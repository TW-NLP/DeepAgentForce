#!/usr/bin/env python
"""验证自定义工具管理 + MCP 配置 CRUD（不起 HTTP，直接测 service 层）。

    PY=/Users/tianwei/miniforge3/envs/agent/bin/python
    $PY scripts/test_tools_mcp_mgmt.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    ok = bool(ok)
    mark = "[PASS]" if ok else "[FAIL]"
    PASS += ok
    FAIL += (not ok)
    print(f"{mark} {name}" + (f"  —— {detail}" if detail else ""))


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="mgmt_test_"))
    data_dir = work / "data"
    data_dir.mkdir()
    tenant = "tenant-aaa"

    # ============ 自定义工具 ============
    print("=" * 70)
    print("自定义工具管理（CustomToolManager）")
    print("=" * 70)
    from src.services.custom_tool_manager import CustomToolManager
    mgr = CustomToolManager(data_dir / "agent_tools_custom")

    good = (
        'def shout(text: str) -> str:\n'
        '    """把文本转成大写并加感叹号。"""\n'
        '    return text.upper() + "!"\n'
    )
    v = mgr.validate_code(good)
    check("合法代码校验通过", v["valid"] and "shout" in v["tool_names"], str(v))

    bad = "def broken(:\n    pass"
    check("语法错误被拒", not mgr.validate_code(bad)["valid"])

    no_doc = "def nodoc(x):\n    return x"
    check("无 docstring/无工具被拒", not mgr.validate_code(no_doc)["valid"])

    r = mgr.save_tool(tenant, "my_tools", good)
    check("保存成功", r["success"] and "shout" in r.get("tool_names", []), str(r))

    r2 = mgr.save_tool(tenant, "my_tools", good)
    check("重名不覆盖（需 force）", not r2["success"], str(r2))
    r3 = mgr.save_tool(tenant, "my_tools", good, force=True)
    check("force 覆盖成功", r3["success"])

    listed = mgr.list_tools(tenant)
    check("列出 1 个文件、含 shout 工具",
          len(listed) == 1 and listed[0]["tools"][0]["name"] == "shout", str(listed))

    loaded = mgr.load_all_tools(tenant)
    check("加载并可调用", loaded and loaded[0].invoke({"text": "hi"}) == "HI!",
          str(loaded[0].invoke({"text": "hi"})) if loaded else "无工具")

    # 租户隔离
    check("其他租户看不到", mgr.list_tools("tenant-bbb") == [])

    d = mgr.delete_tool(tenant, "my_tools")
    check("删除成功", d["success"] and mgr.list_tools(tenant) == [])

    # ============ MCP 配置 ============
    print("\n" + "=" * 70)
    print("MCP 配置 CRUD（McpConfigStore）")
    print("=" * 70)
    import json
    from src.services.mcp_integration import McpConfigStore

    # 全局只读 server
    (data_dir / "mcp_servers.json").write_text(json.dumps({
        "mcpServers": {"global_fs": {"command": "echo", "args": ["x"]}}
    }), encoding="utf-8")

    store = McpConfigStore(data_dir)
    servers = store.list_servers(tenant)
    g = next((s for s in servers if s["name"] == "global_fs"), None)
    check("全局 server 可见且只读",
          g is not None and g["source"] == "global" and not g["editable"], str(g))

    up = store.upsert_server(tenant, "weather", {
        "transport": "streamable_http", "url": "http://localhost:8000/mcp",
    })
    check("新增租户 server", up["success"], str(up))

    up_bad = store.upsert_server(tenant, "bad name!", {"url": "http://x"})
    check("非法名称被拒", not up_bad["success"])

    up_invalid = store.upsert_server(tenant, "empty", {"transport": "stdio"})
    check("既无 command 也无 url 被拒", not up_invalid["success"])

    servers = store.list_servers(tenant)
    w = next((s for s in servers if s["name"] == "weather"), None)
    check("租户 server 可见且可编辑",
          w is not None and w["source"] == "tenant" and w["editable"]
          and w["transport"] == "streamable_http", str(w))

    tog = store.toggle_server(tenant, "weather", enabled=False)
    servers = store.list_servers(tenant)
    w = next((s for s in servers if s["name"] == "weather"), None)
    check("禁用生效", tog["success"] and w and w["enabled"] is False, str(w))

    # 禁用后 collect_mcp_tools 不应连接它
    from src.services.mcp_integration import load_mcp_connections

    class _S:
        DATA_DIR = str(data_dir)

    conns = load_mcp_connections(_S(), tenant)
    check("禁用的 server 不进连接池", "weather" not in conns, str(list(conns)))

    dele = store.delete_server(tenant, "weather")
    check("删除租户 server", dele["success"])
    dele2 = store.delete_server(tenant, "global_fs")
    check("不能删全局 server", not dele2["success"], str(dele2))

    # 租户隔离：别的租户看不到本租户 server
    store.upsert_server(tenant, "secret", {"command": "echo", "args": ["1"]})
    other = [s["name"] for s in store.list_servers("tenant-bbb")]
    check("MCP 租户隔离", "secret" not in other and "global_fs" in other, str(other))

    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
