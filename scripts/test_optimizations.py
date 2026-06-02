#!/usr/bin/env python
"""本会话优化的可用性自测脚本。

覆盖三块：
  1. skills 渐进披露   (src/services/skill_disclosure.py)
  2. 通用工具(15 个)   (src/services/agent_tools/)
  3. tools 渐进披露     (src/services/tool_disclosure.py)
  4. 全链路 agent 构建  (可选，需要一个 LLM 模型字符串)

运行：
    /Users/tianwei/miniforge3/envs/agent/bin/python scripts/test_optimizations.py
    # 跑全链路构建（不会真正请求 LLM，仅验证工具注册）：
    python scripts/test_optimizations.py --full-build

不依赖 pytest；每项打印 [PASS]/[FAIL]，结尾给出汇总，全过返回码 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让脚本能从项目根直接 import（scripts/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 仅用于演示/测试的租户
TEST_TENANT = "test-tenant-0000"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    line = f"[{flag}] {name}"
    if detail:
        line += f"  ——  {detail}"
    print(line)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. skills 渐进披露
# ---------------------------------------------------------------------------


def test_skill_disclosure() -> None:
    section("1. skills 渐进披露 (skill_disclosure.py)")
    from config.settings import settings
    from src.services.skill_disclosure import build_skill_disclosure

    builtin = Path(settings.SKILL_DIR)
    tools, overview = build_skill_disclosure(builtin, None)
    by = {t.name: t for t in tools}

    check("返回 skills_list / skill_view 两个工具",
          set(by) == {"skills_list", "skill_view"},
          f"实际: {sorted(by)}")

    check("概览文本包含分类清单",
          "可用分类" in overview and "渐进式披露" in overview)

    # skills_list 全量
    payload = json.loads(by["skills_list"].func())
    cats = payload.get("categories", [])
    check("skills_list 能列出技能且有分类",
          payload.get("count", 0) > 0 and len(cats) > 0,
          f"技能数={payload.get('count')}, 分类={cats}")

    # 按分类过滤
    if cats:
        first_cat = cats[0]
        sub = json.loads(by["skills_list"].func(category=first_cat))
        ok = all(s["category"] == first_cat for s in sub["skills"]) and sub["count"] > 0
        check(f"skills_list(category='{first_cat}') 过滤正确",
              ok, f"该分类技能数={sub['count']}")

    # skill_view 读全文
    if payload["skills"]:
        sample = payload["skills"][0]["name"]
        body = by["skill_view"].func(name=sample)
        check(f"skill_view('{sample}') 返回 SKILL.md 全文",
              "skill_dir:" in body and len(body) > 100,
              f"长度={len(body)}")

    # 未知技能优雅报错
    err = json.loads(by["skill_view"].func(name="__nope__"))
    check("skill_view 未知技能返回 error", "error" in err)


# ---------------------------------------------------------------------------
# 2. 通用工具（15 个）
# ---------------------------------------------------------------------------


def test_common_tools() -> None:
    section("2. 通用工具 15 个 (agent_tools/)")
    from config.settings import settings
    from src.services.agent_tools import build_common_tools

    tools, overview = build_common_tools(settings, TEST_TENANT)
    by = {t.name: t for t in tools}

    expected = {
        "get_datetime", "calculator", "json_query", "regex_extract", "text_stats",
        "download_file", "list_outputs", "save_text_file", "read_document",
        "web_search", "web_fetch", "http_request",
        "memory_write", "memory_search", "session_search",
    }
    check("共 15 个工具且名称齐全",
          set(by) == expected,
          f"数量={len(by)}; 缺失={expected - set(by)}; 多余={set(by) - expected}")

    # --- 本地工具逐个执行 ---
    r = json.loads(by["get_datetime"].func(tz_offset_hours=8))
    check("get_datetime 返回日期/时间", "date" in r and "time" in r, r.get("iso", ""))

    r = json.loads(by["calculator"].func(expression="(3+4)*2 + sqrt(16)"))
    check("calculator 计算正确", r.get("result") == 18.0, f"result={r.get('result')}")

    r = json.loads(by["calculator"].func(expression="__import__('os')"))
    check("calculator 拒绝危险表达式", "error" in r)

    r = json.loads(by["json_query"].func(
        data='{"items":[{"name":"x"},{"name":"y"}]}', path="items[1].name"))
    check("json_query 取值正确", r.get("value") == "y", str(r))

    r = json.loads(by["regex_extract"].func(text="a1 b22 c333", pattern=r"\d+"))
    check("regex_extract 提取正确", r.get("matches") == ["1", "22", "333"], str(r))

    r = json.loads(by["text_stats"].func(text="你好 world\n第二行"))
    # 中文字符：你好第二行 = 5
    check("text_stats 统计中文/行数", r.get("chinese_chars") == 5 and r.get("lines") == 2, str(r))

    # save -> list -> read_document 闭环
    r = json.loads(by["save_text_file"].func(filename="_selftest.txt", content="hello 自测"))
    saved_ok = r.get("saved_to", "").endswith("_selftest.txt")
    check("save_text_file 保存到租户产物目录", saved_ok, r.get("saved_to", ""))

    r = json.loads(by["list_outputs"].func())
    listed = any(f["name"].endswith("_selftest.txt") for f in r.get("files", []))
    check("list_outputs 能列出刚保存的文件", listed, f"count={r.get('count')}")

    r = json.loads(by["read_document"].func(path="_selftest.txt"))
    check("read_document 读回文本", "hello 自测" in r.get("text", ""), str(r)[:80])

    # save_text_file 越权写应被拒
    r = json.loads(by["save_text_file"].func(filename="../escape.txt", content="x"))
    # 注意：Path(...).name 会把 '../escape.txt' 归一为 'escape.txt'，仍落在产物目录内，
    # 这里只验证不会抛异常且落点安全
    check("save_text_file 路径归一不越权",
          r.get("saved_to", "").endswith("escape.txt") and "outputs" in r.get("saved_to", ""),
          r.get("saved_to", ""))

    # --- 记忆工具闭环 ---
    by["memory_write"].func(content="自测：用户喜欢简洁回答", tags=["selftest"])
    r = json.loads(by["memory_search"].func(query="简洁", limit=5))
    check("memory_write + memory_search 闭环",
          r.get("count", 0) >= 1 and any("简洁" in m["content"] for m in r["memories"]),
          f"命中={r.get('count')}")

    # session_search 不报错（该测试租户通常无历史 -> count 0 也算通过）
    r = json.loads(by["session_search"].func(query="你好", limit=3))
    check("session_search 可执行", "matches" in r, f"命中={r.get('count')}")

    # --- 联网工具：仅检查密钥与可调用性，不强制联网 ---
    has_tavily = bool(getattr(settings, "TAVILY_API_KEY", ""))
    if has_tavily:
        r = json.loads(by["web_search"].func(query="OpenAI", max_results=2))
        check("web_search 实联网返回结果", r.get("count", 0) >= 0 and "error" not in r, str(r)[:80])
    else:
        r = json.loads(by["web_search"].func(query="x"))
        check("web_search 无密钥时优雅报错(未配置 TAVILY_API_KEY)", "error" in r)

    # 清理自测产物
    _cleanup_test_artifacts(settings)


def _cleanup_test_artifacts(settings) -> None:
    try:
        out = Path(settings.get_tenant_output_dir(TEST_TENANT))
        for fn in ("_selftest.txt", "escape.txt"):
            (out / fn).unlink(missing_ok=True)
        mem = Path(settings.DATA_DIR) / "agent_memory" / f"{TEST_TENANT}.json"
        mem.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 3. tools 渐进披露
# ---------------------------------------------------------------------------


def _dummy_tool(name: str, desc: str):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _Args(BaseModel):
        x: str = Field(default="", description="参数 x")

    return StructuredTool.from_function(
        func=lambda x="", _n=name: f"{_n} ran with {x!r}",
        name=name, description=desc, args_schema=_Args,
    )


def test_tool_disclosure() -> None:
    section("3. tools 渐进披露 (tool_disclosure.py)")
    from src.services.tool_disclosure import (
        build_tool_disclosure, estimate_tokens, should_activate,
    )

    # 空池 -> 无操作
    tools, ov = build_tool_disclosure([], context_length=None)
    check("空池返回空工具且概览为空", tools == [] and ov == "")

    # 小池 -> 直接绑定
    small = [_dummy_tool(f"t{i}", "短描述") for i in range(3)]
    tools, ov = build_tool_disclosure(small, context_length=200_000)
    check("小池直接绑定(不切桥接)",
          {t.name for t in tools} == {"t0", "t1", "t2"} and ov == "")

    # 大池 -> 切换桥接工具
    big = [
        _dummy_tool("send_email", "发送电子邮件给收件人，支持抄送附件 send email message recipients"),
        _dummy_tool("create_github_issue", "在 github 仓库创建 issue，含标题正文 create github issue"),
        _dummy_tool("weather_lookup", "查询城市实时天气温度 weather forecast temperature city"),
    ]
    big += [_dummy_tool(f"dummy_{i}", "占位工具 " * 20) for i in range(320)]
    toks = estimate_tokens(big)
    check("大池 token 估算 + should_activate 触发",
          should_activate(toks, None) is True, f"est_tokens={toks}")

    btools, bov = build_tool_disclosure(big, context_length=None)
    bset = {t.name for t in btools}
    check("大池切换为 3 个桥接工具",
          bset == {"tool_search", "tool_describe", "tool_invoke"}, f"实际={sorted(bset)}")
    check("披露概览文本生成", "渐进式披露" in bov and "tool_search" in bov)

    by = {t.name: t for t in btools}

    # BM25 检索（中文 query）
    r = json.loads(by["tool_search"].func(query="发送邮件 email", limit=3))
    names = [t["name"] for t in r["tools"]]
    check("tool_search('发送邮件') 命中 send_email", "send_email" in names, f"top={names}")

    r = json.loads(by["tool_search"].func(query="github issue", limit=3))
    names = [t["name"] for t in r["tools"]]
    check("tool_search('github issue') 命中 create_github_issue",
          "create_github_issue" in names, f"top={names}")

    # describe
    r = json.loads(by["tool_describe"].func(name="send_email"))
    check("tool_describe 返回参数 schema", "parameters" in r and r.get("name") == "send_email")

    # invoke 分发执行
    r = by["tool_invoke"].func(name="send_email", args={"x": "hi"})
    check("tool_invoke 正确分发并执行", "send_email ran with" in str(r), str(r)[:60])

    # 未知工具
    r = json.loads(by["tool_invoke"].func(name="__nope__", args={}))
    check("tool_invoke 未知工具优雅报错", "error" in r)


# ---------------------------------------------------------------------------
# 4. 全链路 agent 构建（可选）
# ---------------------------------------------------------------------------


def test_full_build(model: str) -> None:
    section(f"4. 全链路 agent 构建 (model={model})")
    import os
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
    from config.settings import settings
    settings.LLM_MODEL = model
    from src.services.conversational_agent import ConversationalAgent

    agent = ConversationalAgent(settings=settings, tenant_uuid=TEST_TENANT)
    inst = agent.build_instance()
    tn = inst.nodes.get("tools")
    tbn = getattr(getattr(tn, "bound", tn), "tools_by_name", None) or getattr(tn, "tools_by_name", {})
    names = set(tbn.keys())

    check("agent 成功 build", inst is not None)
    check("skills 工具已注册", {"skills_list", "skill_view"} <= names)
    check("通用工具已注册", {"calculator", "web_search", "memory_write"} <= names)
    # extra_tools 当前为空 -> 不应出现桥接工具
    check("空额外池下无桥接工具(符合预期)",
          not ({"tool_search", "tool_invoke"} & names),
          f"工具总数={len(names)}")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-build", action="store_true", help="额外跑全链路 agent 构建")
    parser.add_argument("--model", default="openai:gpt-4o-mini",
                        help="全链路构建用的模型字符串（不会真正请求）")
    args = parser.parse_args()

    test_skill_disclosure()
    test_common_tools()
    test_tool_disclosure()
    if args.full_build:
        test_full_build(args.model)

    section("汇总")
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, _ in _results:
        if not ok:
            print(f"  ✗ {name}")
    print(f"\n{passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
