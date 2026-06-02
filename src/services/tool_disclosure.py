"""额外 / MCP 工具的渐进式披露（progressive tool disclosure）。

与 skills 的渐进披露同源，但有一个本质差别：**skill 只是文本，tool 是可调用对象**。
deepagents/LangGraph 在编译时把工具列表静态绑定到模型，无法中途再把某个工具绑上去，
因此 tool 的「第二步」不是「读全文」，而是「真正执行」——需要一个分发层。

做法（参照 hermes tools/tool_search.py）：

- **必备工具**（deepagents 内置 + 本项目通用工具 + skills 工具）永远直接绑定，绝不延迟。
- **额外 / MCP 工具**先进一个 registry（留在内存，schema 不进上下文）。
- 阈值门控：估算这批工具 schema 的 token 占用，仅当其 ≥ 上下文的 ``threshold_pct``
  （默认 10%）时才启用披露；否则直接绑定，不值得绕一层。
- 启用披露后，模型可见的只有三个桥接工具：
    * ``tool_search(query)``  —— BM25 检索，返回匹配工具的 name+description（菜单层）；
    * ``tool_describe(name)`` —— 返回该工具的完整参数 schema（全文层）；
    * ``tool_invoke(name, args)`` —— 按名字从 registry 取出真实工具并执行（执行层）。

无论后面挂 10 个还是 500 个 MCP 工具，常驻上下文的开销恒为这三个桥接工具。
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 无真实 tokenizer 时按 chars/4 估算 token，跨厂商稳定（与 hermes 一致）。
CHARS_PER_TOKEN = 4.0
# 上下文未知时的固定 token 阈值兜底（Anthropic/OpenAI 都在此量级观察到质量下降）。
_FALLBACK_TOKEN_CUTOFF = 20_000

TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_INVOKE_NAME = "tool_invoke"
BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_INVOKE_NAME})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")

_jieba_mod = None
_jieba_tried = False


def _run_coro(coro):
    """同步执行协程，兼容已有运行中的事件循环（另起线程跑独立 loop）。"""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def _err(message: str, **extra: Any) -> str:
    payload = {"error": str(message)}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 分词（中英混合）
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """中英混合分词：拉丁词/数字用正则，中文用 jieba（不可用时退化为单字）。"""
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    cjk_present = bool(_CJK_RE.search(text))
    if not cjk_present:
        return tokens

    global _jieba_mod, _jieba_tried
    if not _jieba_tried:
        _jieba_tried = True
        try:
            import jieba  # 延迟导入，首次较慢
            _jieba_mod = jieba
        except Exception:
            _jieba_mod = None

    if _jieba_mod is not None:
        for seg in _jieba_mod.lcut(text):
            seg = seg.strip()
            if seg and _CJK_RE.search(seg):
                tokens.append(seg)
    else:
        tokens.extend(_CJK_RE.findall(text))
    return tokens


# ---------------------------------------------------------------------------
# 工具 schema 提取与 token 估算
# ---------------------------------------------------------------------------


def _tool_param_schema(tool: BaseTool) -> Dict[str, Any]:
    """返回工具的参数 properties dict（尽量复用 LangChain 的 args）。"""
    try:
        return dict(tool.args or {})
    except Exception:
        return {}


def _tool_openai_like(tool: BaseTool) -> Dict[str, Any]:
    """构造一个近似 OpenAI function 的 schema dict，仅用于 token 估算与 describe。"""
    props = _tool_param_schema(tool)
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": {"type": "object", "properties": props},
    }


def estimate_tokens(tools: List[BaseTool]) -> int:
    """按 chars/4 估算一批工具 schema 的 token 占用。"""
    total_chars = 0
    for t in tools:
        try:
            total_chars += len(
                json.dumps(_tool_openai_like(t), ensure_ascii=False, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            total_chars += len(str(t.name) + str(t.description))
    return int(math.ceil(total_chars / CHARS_PER_TOKEN))


def should_activate(
    deferrable_tokens: int,
    context_length: Optional[int],
    threshold_pct: float = 10.0,
    mode: str = "auto",
) -> bool:
    """是否启用披露。mode: 'auto'|'on'|'off'。"""
    if mode == "off":
        return False
    if deferrable_tokens <= 0:
        return False
    if mode == "on":
        return True
    if not context_length or context_length <= 0:
        return deferrable_tokens >= _FALLBACK_TOKEN_CUTOFF
    threshold_tokens = int(context_length * (threshold_pct / 100.0))
    return deferrable_tokens >= threshold_tokens


# ---------------------------------------------------------------------------
# 工具目录 + BM25 检索
# ---------------------------------------------------------------------------


@dataclass
class _CatalogEntry:
    name: str
    description: str
    tool: BaseTool
    _tokens: List[str] = field(default_factory=list)


def _entry_search_text(tool: BaseTool) -> str:
    """检索文本：工具名（拆词）+ 描述 + 参数名。schema 主体不入索引（噪声大、收益低）。"""
    name_words = re.sub(r"[_.:\-]+", " ", tool.name)
    param_names = " ".join(_tool_param_schema(tool).keys())
    return f"{name_words} {tool.description or ''} {param_names}"


class ToolCatalog:
    """延迟工具的目录：支持 BM25 检索、describe、按名取真实工具执行。"""

    def __init__(self, tools: List[BaseTool]):
        self._by_name: Dict[str, BaseTool] = {}
        self._entries: List[_CatalogEntry] = []
        for t in tools:
            if t.name in self._by_name:
                logger.warning("延迟工具重名，后者覆盖前者: %s", t.name)
            self._by_name[t.name] = t
            self._entries.append(
                _CatalogEntry(
                    name=t.name,
                    description=t.description or "",
                    tool=t,
                    _tokens=_tokenize(_entry_search_text(t)),
                )
            )

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._by_name.get(name)

    def search(self, query: str, limit: int = 5) -> List[_CatalogEntry]:
        if not self._entries or limit <= 0:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        doc_lengths = [len(e._tokens) for e in self._entries]
        avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
        doc_freq: Dict[str, int] = {}
        for e in self._entries:
            for tok in set(e._tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        n_docs = len(self._entries)

        scored: List[Tuple[float, _CatalogEntry]] = []
        for e in self._entries:
            s = _bm25_score(query_tokens, e._tokens, avg_dl, doc_freq, n_docs)
            if s > 0:
                scored.append((s, e))

        if not scored:
            # 子串兜底：BM25 在「所有文档都含同一词」时 IDF 为 0，需补充。
            ql = query.lower()
            for e in self._entries:
                if ql in e.name.lower() or ql in e.description.lower():
                    scored.append((0.1, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_dl: float,
    doc_freq: Dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    doc_tf: Dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    score = 0.0
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


# ---------------------------------------------------------------------------
# 桥接工具
# ---------------------------------------------------------------------------


class _SearchArgs(BaseModel):
    query: str = Field(description="用自然语言或关键词描述你需要的能力，如 '发送邮件' 或 'github issue'。")
    limit: int = Field(default=5, description="返回的候选工具数量（1-20）。")


class _DescribeArgs(BaseModel):
    name: str = Field(description="要查看完整参数说明的工具名（来自 tool_search 结果）。")


class _InvokeArgs(BaseModel):
    name: str = Field(description="要调用的工具名（来自 tool_search 结果）。")
    args: Optional[dict] = Field(default=None, description="传给该工具的参数对象（键值对）。")


def _make_bridge_tools(catalog: ToolCatalog) -> List[StructuredTool]:
    def _search(query: str, limit: int = 5) -> str:
        hits = catalog.search(query, max(1, min(limit, 20)))
        return _ok({
            "count": len(hits),
            "tools": [{"name": e.name, "description": e.description} for e in hits],
            "hint": "用 tool_describe(name) 看参数，再用 tool_invoke(name, args) 调用",
        })

    def _describe(name: str) -> str:
        tool = catalog.get(name)
        if tool is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search 检索")
        return _ok({
            "name": tool.name,
            "description": tool.description or "",
            "parameters": _tool_param_schema(tool),
        })

    def _invoke(name: str, args: Optional[dict] = None) -> str:
        tool = catalog.get(name)
        if tool is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search 检索")
        try:
            result = tool.invoke(args or {})
            return result if isinstance(result, str) else _ok(result)
        except NotImplementedError:
            # MCP 等异步专用工具不支持同步 invoke，退回异步执行。
            try:
                result = _run_coro(tool.ainvoke(args or {}))
                return result if isinstance(result, str) else _ok(result)
            except Exception as e:  # noqa: BLE001
                return _err(f"调用工具 '{name}' 失败: {e}")
        except Exception as e:  # noqa: BLE001
            return _err(f"调用工具 '{name}' 失败: {e}")

    async def _ainvoke(name: str, args: Optional[dict] = None) -> str:
        # Agent 经 LangGraph 异步执行时走这里；MCP 异步工具天然契合。
        tool = catalog.get(name)
        if tool is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search 检索")
        try:
            result = await tool.ainvoke(args or {})
            return result if isinstance(result, str) else _ok(result)
        except Exception as e:  # noqa: BLE001
            return _err(f"调用工具 '{name}' 失败: {e}")

    return [
        StructuredTool.from_function(
            func=_search, name=TOOL_SEARCH_NAME,
            description=(
                "在「额外/MCP 工具库」中按需检索可用工具（返回 name+description）。"
                "当必备工具无法满足、可能需要某个外部/集成能力时先调用它。"
            ),
            args_schema=_SearchArgs,
        ),
        StructuredTool.from_function(
            func=_describe, name=TOOL_DESCRIBE_NAME,
            description="查看某个工具的完整参数 schema；在 tool_search 选定后、调用前使用。",
            args_schema=_DescribeArgs,
        ),
        StructuredTool.from_function(
            func=_invoke, coroutine=_ainvoke, name=TOOL_INVOKE_NAME,
            description="按名字执行额外/MCP 工具并返回结果。args 为该工具所需的参数对象。",
            args_schema=_InvokeArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def build_tool_disclosure(
    extra_tools: List[BaseTool],
    context_length: Optional[int] = None,
    threshold_pct: float = 10.0,
    mode: str = "auto",
) -> Tuple[List[BaseTool], str]:
    """构建额外/MCP 工具的披露层。

    返回 ``(要绑定到模型的工具, system-prompt 概览文本)``：

    - 当额外工具池小于阈值（或 mode='off'）：直接返回这些工具本身（全部绑定），概览为空。
    - 当达到阈值（或 mode='on'）：返回三个桥接工具，概览说明两步用法。
    """
    if not extra_tools:
        return [], ""

    tokens = estimate_tokens(extra_tools)
    if not should_activate(tokens, context_length, threshold_pct, mode):
        logger.info(
            "额外工具池 %d 个（约 %d tokens）未达阈值，直接绑定。", len(extra_tools), tokens
        )
        return list(extra_tools), ""

    catalog = ToolCatalog(extra_tools)
    logger.info(
        "额外工具池 %d 个（约 %d tokens）达阈值，启用渐进披露（tool_search/describe/invoke）。",
        len(catalog), tokens,
    )
    overview = (
        "## 额外工具库（渐进式披露）\n\n"
        f"另有 **{len(catalog)}** 个额外/MCP 工具未直接列出，按需两步取用：\n"
        "1. `tool_search(query)` 按能力检索，拿到候选工具的 name+description；\n"
        "2. （可选）`tool_describe(name)` 看参数，再 `tool_invoke(name, args)` 执行。\n\n"
        "必备工具能直接满足时无需走这一层。"
    )
    return _make_bridge_tools(catalog), overview
