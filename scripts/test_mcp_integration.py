#!/usr/bin/env python
"""端到端验证 MCP 接入：起一个本地 stdio MCP server，确认工具被发现、
经渐进披露层处理、并可真正调用。

不依赖任何外部服务，自带一个最小 MCP server（mcp_dummy_server.py，运行时生成）。

用法：
    PY=/Users/tianwei/miniforge3/envs/agent/bin/python
    $PY scripts/test_mcp_integration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"{mark} {name}" + (f"  —— {detail}" if detail else ""))


# 一个最小 MCP server（stdio），暴露两个工具：add / echo
_DUMMY_SERVER = '''#!/usr/bin/env python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dummy")


@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入文本。"""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def main() -> int:
    py = sys.executable
    workdir = Path(tempfile.mkdtemp(prefix="mcp_test_"))
    server_path = workdir / "mcp_dummy_server.py"
    server_path.write_text(_DUMMY_SERVER, encoding="utf-8")

    # 用一个隔离的 data 目录承载 mcp_servers.json，避免污染真实配置
    data_dir = workdir / "data"
    data_dir.mkdir()
    (data_dir / "mcp_servers.json").write_text(
        json.dumps({
            "mcpServers": {
                "dummy": {
                    "command": py,
                    "args": [str(server_path)],
                    "transport": "stdio",
                },
                "disabled_one": {
                    "command": py,
                    "args": [str(server_path)],
                    "disabled": True,
                },
            }
        }),
        encoding="utf-8",
    )

    class _Settings:
        DATA_DIR = str(data_dir)

    settings = _Settings()

    print("=" * 70)
    print("1) 配置加载与规范化")
    print("=" * 70)
    from src.services.mcp_integration import load_mcp_connections, collect_mcp_tools

    conns = load_mcp_connections(settings, tenant_uuid=None)
    check("只保留启用的 server", set(conns) == {"dummy"}, f"得到 {set(conns)}")
    check("stdio transport 正确", conns.get("dummy", {}).get("transport") == "stdio")
    check("command 透传正确", conns.get("dummy", {}).get("command") == py)

    print("\n" + "=" * 70)
    print("2) 连接 server 并发现工具（真实子进程）")
    print("=" * 70)
    tools = collect_mcp_tools(settings, tenant_uuid=None)
    names = {t.name for t in tools}
    check("发现两个工具", len(tools) == 2, f"得到 {len(tools)}: {names}")
    check("工具名带 mcp__dummy__ 前缀",
          names == {"mcp__dummy__add", "mcp__dummy__echo"}, f"{names}")

    add_tool = next((t for t in tools if t.name.endswith("__add")), None)
    echo_tool = next((t for t in tools if t.name.endswith("__echo")), None)
    check("工具携带 description",
          bool(add_tool and add_tool.description), )

    print("\n" + "=" * 70)
    print("3) 真正调用 MCP 工具（MCP 工具为异步专用，走 ainvoke）")
    print("=" * 70)
    import asyncio

    if add_tool is not None:
        try:
            res = asyncio.run(add_tool.ainvoke({"a": 123, "b": 456}))
            check("add(123,456) == 579", "579" in str(res), f"得到 {res!r}")
        except Exception as e:
            check("add 调用成功", False, str(e))
    if echo_tool is not None:
        try:
            res = asyncio.run(echo_tool.ainvoke({"text": "你好"}))
            check("echo 回显正确", "你好" in str(res), f"得到 {res!r}")
        except Exception as e:
            check("echo 调用成功", False, str(e))

    print("\n" + "=" * 70)
    print("4) 经 build_tool_disclosure 处理（少量工具应直接绑定）")
    print("=" * 70)
    from src.services.tool_disclosure import build_tool_disclosure
    bound, overview = build_tool_disclosure(tools, context_length=None)
    check("少量 MCP 工具直接绑定（不走桥接）",
          {t.name for t in bound} == names and overview == "",
          f"bound={len(bound)} overview_empty={overview==''}")

    print("\n" + "=" * 70)
    print("5) 工具池放大后应触发渐进披露（桥接工具）")
    print("=" * 70)
    # 复制放大到模拟「上百个 MCP 工具」，强制达到 20k token 兜底阈值
    big_pool = list(tools)
    from langchain_core.tools import StructuredTool

    def _noop(x: str = "") -> str:
        return x

    for i in range(400):
        big_pool.append(StructuredTool.from_function(
            func=_noop, name=f"mcp__bulk__tool_{i}",
            description="一个用于压测渐进披露阈值的占位 MCP 工具，描述要足够长以累积 token 占用。" * 2,
        ))
    bridged, ov = build_tool_disclosure(big_pool, context_length=None)
    bnames = {t.name for t in bridged}
    # 纯 MCP 池 → 走 mcp_search（三层服务级），无自定义工具故无 tool_search。
    check("大池子切换为桥接工具(MCP→mcp_search)",
          bnames == {"mcp_search", "tool_describe", "tool_invoke"}, f"{bnames}")
    check("概览文本非空", bool(ov))

    print("\n" + "=" * 70)
    print("6) mcp_search 服务级检索 + 桥接 tool_invoke 调用异步专用 MCP 工具")
    print("=" * 70)
    # 把真实 MCP 工具放进披露层（强制 mode='on' 走桥接），验证服务级检索与调用打通
    bridged2, _ = build_tool_disclosure(tools, context_length=None, mode="on")
    search_tool = next(t for t in bridged2 if t.name == "mcp_search")
    invoke_tool = next(t for t in bridged2 if t.name == "tool_invoke")

    res = json.loads(search_tool.invoke({"query": "求和 加法 add"}))
    hit_tool_names = {tt["name"] for s in res.get("services", []) for tt in s.get("tools", [])}
    check("mcp_search 命中含 add 的服务",
          "mcp__dummy__add" in hit_tool_names, f"{hit_tool_names}")

    # 同步入口（内部 NotImplementedError → 退回 ainvoke）
    sync_res = invoke_tool.invoke({"name": "mcp__dummy__add", "args": {"a": 7, "b": 8}})
    check("tool_invoke 同步入口调用 MCP 成功", "15" in str(sync_res), f"得到 {sync_res!r}")

    # 异步入口（Agent 实际走的路径）
    async_res = asyncio.run(
        invoke_tool.ainvoke({"name": "mcp__dummy__echo", "args": {"text": "嗨"}})
    )
    check("tool_invoke 异步入口调用 MCP 成功", "嗨" in str(async_res), f"得到 {async_res!r}")

    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
