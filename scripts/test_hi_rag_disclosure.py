#!/usr/bin/env python
"""Hi-RAG 分层工具披露自测。

覆盖：
  1. Type 类目表（tool_taxonomy）归一化/反查
  2. 两入口拆池：自定义→tool_search(2层)、MCP→mcp_search(3层)、共用 describe/invoke
  3. mcp_search 三层：Tool-as-Proxy 粗排 → 上卷服务 → 返回服务及工具清单（带 type/描述）
  4. 混合检索：假 embedder 下「词法不相交但语义相关」的工具能被粗排召回；
     无 embedder 时退回纯 BM25（同一查询召回不到），证明向量路确实生效
  5. 细排 _rerank 在 embedder 下按语义重排
  6. Type 端到端：custom_tool_manager(_meta.json) 与 McpConfigStore 的 type 透传

运行：
    /Users/tianwei/miniforge3/envs/agent/bin/python scripts/test_hi_rag_disclosure.py

不依赖 pytest；每项打印 [PASS]/[FAIL]，全过返回码 0。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.tools import StructuredTool  # noqa: E402

from src.services.tool_disclosure import (  # noqa: E402
    BRIDGE_TOOL_NAMES,
    _Hybrid,
    _build_entry,
    _rerank,
    build_tool_disclosure,
)
from src.services.tool_taxonomy import (  # noqa: E402
    TOOL_TYPES,
    normalize_type,
    type_label,
)

TEST_TENANT = "test-hirag-0000"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}" + (f"  ——  {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _mk(name: str, desc: str, metadata: dict | None = None) -> StructuredTool:
    t = StructuredTool.from_function(func=lambda x="": x, name=name, description=desc)
    if metadata is not None:
        t.metadata = metadata
    return t


# 假 embedder：按「主题触发词」把文本映射到固定维向量，确定性、无网络。
# 关键：主题里放同义词（pull/merge/pr/request 同维），使语义相关但词法不同的文本向量相近。
_TOPICS = [
    {"pull", "merge", "pr", "request", "合并"},
    {"mail", "email", "邮件"},
    {"search", "bing", "搜索"},
    {"weather", "天气"},
]


class FakeEmbedder:
    available = True

    async def aembed(self, texts):
        out = []
        for t in texts:
            tl = str(t).lower()
            out.append([float(sum(1 for w in topic if w in tl)) for topic in _TOPICS])
        return out


# ---------------------------------------------------------------------------
# 1. Type 类目表
# ---------------------------------------------------------------------------
def test_taxonomy() -> None:
    section("1. Type 类目表 (tool_taxonomy)")
    slugs = {t["slug"] for t in TOOL_TYPES}
    check("固定为 Hi-RAG 8 类", len(TOOL_TYPES) == 8 and "file-systems" in slugs, f"{sorted(slugs)}")
    check("大小写/空白归一化", normalize_type("  FILE-SYSTEMS ") == "file-systems")
    check("中文 label 反查 slug", normalize_type("搜索") == "search")
    check("非法/缺失 → 空 Type", normalize_type("nonsense") == "" and normalize_type("") == "")
    check("slug → 中文 label", type_label("research") == "研究")


# ---------------------------------------------------------------------------
# 2 & 3. 拆池 + 两入口 + mcp_search 三层
# ---------------------------------------------------------------------------
def test_split_and_search() -> None:
    section("2&3. 拆池 / tool_search(2层) / mcp_search(3层)")
    tools = [
        _mk("mcp__github__create_pr", "create a pull request on github 创建 PR",
            {"mcp_service": "github", "mcp_type": "browser", "mcp_service_description": "GitHub 服务"}),
        _mk("mcp__github__create_issue", "create github issue 创建议题",
            {"mcp_service": "github", "mcp_type": "browser", "mcp_service_description": "GitHub 服务"}),
        _mk("mcp__bing__web_search", "search the web via bing 网络搜索",
            {"mcp_service": "bing", "mcp_type": "search", "mcp_service_description": "Bing 搜索"}),
        _mk("send_email", "发送电子邮件 send email to recipient", {"custom_type": "calendar"}),
        _mk("add_numbers", "两个数字相加 add two numbers", {"custom_type": ""}),
    ]
    bridges, overview = build_tool_disclosure(tools, mode="on", settings=None)
    by = {t.name: t for t in bridges}
    check("混合池 → 四个桥接工具", set(by) == BRIDGE_TOOL_NAMES, f"{sorted(by)}")
    check("概览含 tool_search/mcp_search/渐进式披露",
          all(s in overview for s in ("tool_search", "mcp_search", "渐进式披露")))

    r = json.loads(by["tool_search"].func(query="发送邮件 email", limit=3))
    names = [(t["name"], t["type"]) for t in r["tools"]]
    check("tool_search 命中自定义 send_email 且带 type",
          ("send_email", "calendar") in names, f"{names}")

    r = json.loads(by["mcp_search"].func(query="github pull request 创建 PR", limit=3))
    gh = next((s for s in r["services"] if s["service"] == "github"), None)
    check("mcp_search 返回服务级结果(github)", gh is not None, f'{[s["service"] for s in r["services"]]}')
    check("服务带 type + 描述", gh and gh["type"] == "browser" and "GitHub" in gh["description"])
    check("服务内含其全部工具(上卷)",
          gh and {"mcp__github__create_pr", "mcp__github__create_issue"}
          <= {t["name"] for t in gh["tools"]})

    r = json.loads(by["tool_describe"].func(name="mcp__bing__web_search"))
    check("describe 共用且带 service/type", r.get("service") == "bing" and r.get("type") == "search")
    out = by["tool_invoke"].func(name="add_numbers", args={"x": "hi"})
    check("invoke 共用并执行", "hi" in str(out), str(out)[:40])

    # 纯自定义池 → 只暴露 tool_search（无 mcp_search）
    only_custom, _ = build_tool_disclosure(
        [_mk(f"c{i}", "占位 " * 5, {"custom_type": ""}) for i in range(3)], mode="on", settings=None)
    cnames = {t.name for t in only_custom}
    check("纯自定义池不暴露 mcp_search", "mcp_search" not in cnames and "tool_search" in cnames, f"{sorted(cnames)}")


# ---------------------------------------------------------------------------
# 4 & 5. 混合检索向量路 + 细排
# ---------------------------------------------------------------------------
def test_hybrid_and_rerank() -> None:
    section("4&5. 混合检索(向量路) + 细排 _rerank")
    # doc_pr 与查询「合并我最新的改动」词法零交集，但语义相关（同主题：合并/pull/request）。
    e_pr = _build_entry(_mk("open_pr", "open a pull request on the repository"))
    e_weather = _build_entry(_mk("weather_lookup", "check the weather forecast"))
    query = "把我最新的改动 合并"  # tokens: 把/我/最新/改动/合并；与 e_pr 描述无共同词

    # 纯 BM25（embedder 不可用）：召回不到 e_pr
    bm25_only = _Hybrid([e_pr, e_weather], SimpleNamespace(available=False, aembed=None))
    cand_bm25, _ = asyncio.run(bm25_only.coarse(query, 5))
    check("纯 BM25 召回不到词法不相交的 e_pr",
          all(e.name != "open_pr" for e in cand_bm25), f"{[e.name for e in cand_bm25]}")

    # 混合（假 embedder）：向量路把语义相关的 e_pr 召回
    hybrid = _Hybrid([e_pr, e_weather], FakeEmbedder())
    cand, q_vec = asyncio.run(hybrid.coarse(query, 5))
    check("混合检索(向量路)召回 e_pr", any(e.name == "open_pr" for e in cand), f"{[e.name for e in cand]}")
    check("coarse 回传 query 向量供细排复用", q_vec is not None)

    # 细排：对候选按语义重排，e_pr（同主题）应排在 e_weather 前
    cand2 = [e_pr, e_weather]
    scores = asyncio.run(_rerank(q_vec, [e.fine_text for e in cand2], FakeEmbedder()))
    check("_rerank 产出与候选对齐的分数", scores is not None and len(scores) == 2, f"{scores}")
    check("细排把语义相关 e_pr 排前", scores and scores[0] > scores[1], f"{scores}")

    # embedder 不可用时 _rerank 返回 None（保持粗排序）
    none_scores = asyncio.run(_rerank(q_vec, ["a", "b"], SimpleNamespace(available=False, aembed=None)))
    check("无 embedder 时 _rerank 返回 None", none_scores is None)


# ---------------------------------------------------------------------------
# 6. Type 端到端（custom_tool_manager + McpConfigStore）
# ---------------------------------------------------------------------------
def test_type_plumbing() -> None:
    section("6. Type 端到端透传")
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)

        # --- 自定义工具：save 带 type → _meta.json → list/load 透传 ---
        from src.services.custom_tool_manager import CustomToolManager
        mgr = CustomToolManager(data_dir / "agent_tools_custom")
        code = '''
def shout(text: str) -> str:
    """把文本转成大写并加感叹号。"""
    return text.upper() + "!"
'''
        res = mgr.save_tool(TEST_TENANT, "my_tool", code, tool_type="SEARCH")
        check("save_tool 归一化并回传 type=search", res.get("success") and res.get("type") == "search", str(res)[:80])
        listed = mgr.list_tools(TEST_TENANT)
        check("list_tools 带回 type", listed and listed[0].get("type") == "search", str(listed)[:80])
        loaded = mgr.load_all_tools(TEST_TENANT)
        md = (getattr(loaded[0], "metadata", None) or {}) if loaded else {}
        check("load_all_tools 给代理工具附 custom_type", md.get("custom_type") == "search", f"{md}")

        # --- MCP：upsert 带 type/description → list 归一化 + load_mcp_meta ---
        from src.services.mcp_integration import McpConfigStore, load_mcp_meta
        store = McpConfigStore(data_dir)
        up = store.upsert_server(TEST_TENANT, "bing", {
            "transport": "streamable_http", "url": "http://localhost/mcp",
            "type": "搜索", "description": "Bing 搜索服务",
        })
        check("upsert MCP server 成功", up.get("success"), str(up)[:60])
        info = next((s for s in store.list_servers(TEST_TENANT) if s["name"] == "bing"), None)
        check("list_servers 归一化 type=search + 描述",
              info and info["type"] == "search" and info["description"] == "Bing 搜索服务", str(info)[:120])

        settings = SimpleNamespace(DATA_DIR=data_dir, MCP_SERVERS=None)
        meta = load_mcp_meta(settings, TEST_TENANT)
        check("load_mcp_meta 读出 type/描述",
              meta.get("bing", {}).get("type") == "search"
              and "Bing" in meta.get("bing", {}).get("description", ""), f"{meta}")


def main() -> int:
    test_taxonomy()
    test_split_and_search()
    test_hybrid_and_rerank()
    test_type_plumbing()
    section("汇总")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n{passed}/{len(_results)} 通过" + (f"，{failed} 失败" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
