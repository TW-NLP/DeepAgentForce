"""额外 / MCP 工具的渐进式披露（progressive tool disclosure）。

参照 **Hi-RAG**（A Hierarchical Framework for Scalable and Generalizable Tool
Selection）把工具检索从「扁平 BM25」升级为 **Type → Service → Tool 分层、
粗粒度检索 + 细粒度重排**：

- **粗排（coarse）**：混合检索 = BM25(词法) + embedding(语义) + 加权 RRF 融合
  （k=60, α=0.1，偏向语义；与论文一致）。未配置 embedding 时自动退回纯 BM25。
- **细排（fine）**：对粗排候选用 embedding 余弦相似度重排。

两类额外工具走**两套独立**披露，但共用 `tool_describe` / `tool_invoke`：

- **自定义工具**（无 service）→ `tool_search(query)`：两层 Type→Tool。
  粗排用工具信息，细排用 ``type + 工具描述``。
- **MCP 工具**（有 service=server）→ `mcp_search(query)`：三层 Type→Service→Tool。
  粗排用工具信息（Tool-as-Proxy：检索工具→上卷到父服务），细排用
  ``type + 服务描述 + 旗下各工具描述`` 重排服务，返回候选服务及其工具清单。

与 skills 的渐进披露同源，但 tool 是可调用对象：第二步不是「读全文」而是
「真正执行」，故 `tool_invoke` 提供分发层。无论挂多少工具，常驻上下文只有
这几个桥接工具。

阈值门控（沿用）：额外工具池 schema token 占用 < 上下文 ``threshold_pct``
（默认 10%）时直接绑定，不值得绕一层；达阈值才启用桥接。
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

from src.services.tool_taxonomy import normalize_type, type_label

logger = logging.getLogger(__name__)

# 无真实 tokenizer 时按 chars/4 估算 token，跨厂商稳定。
CHARS_PER_TOKEN = 4.0
# 上下文未知时的固定 token 阈值兜底。
_FALLBACK_TOKEN_CUTOFF = 20_000

# 混合检索参数（与 Hi-RAG 论文一致）：RRF 平滑因子 k，权重 α（偏向语义检索）。
_RRF_K = 60
_RRF_ALPHA = 0.1

TOOL_SEARCH_NAME = "tool_search"
MCP_SEARCH_NAME = "mcp_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_INVOKE_NAME = "tool_invoke"
BRIDGE_TOOL_NAMES = frozenset(
    {TOOL_SEARCH_NAME, MCP_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_INVOKE_NAME}
)

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
# 检索文本与 BM25
# ---------------------------------------------------------------------------


def _entry_search_text(tool: BaseTool) -> str:
    """粗排检索文本：工具名（拆词）+ 描述 + 参数名。schema 主体不入索引。"""
    name_words = re.sub(r"[_.:\-]+", " ", tool.name)
    param_names = " ".join(_tool_param_schema(tool).keys())
    return f"{name_words} {tool.description or ''} {param_names}"


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


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _ranks(scored: Dict[int, float]) -> Dict[int, int]:
    """score 字典 → 名次字典（1 起，分高名次小）。"""
    ordered = sorted(scored, key=lambda i: scored[i], reverse=True)
    return {idx: rank for rank, idx in enumerate(ordered, start=1)}


def _rrf_fuse(
    sparse: Dict[int, float], dense: Dict[int, float], n: int
) -> Dict[int, float]:
    """加权 RRF 融合两路名次。dense 为空时退化为纯 BM25 名次。"""
    sr = _ranks(sparse)
    dr = _ranks(dense)
    has_dense = bool(dense)
    fused: Dict[int, float] = {}
    for i in range(n):
        s = 0.0
        if has_dense:
            if i in sr:
                s += _RRF_ALPHA / (_RRF_K + sr[i])
            if i in dr:
                s += (1 - _RRF_ALPHA) / (_RRF_K + dr[i])
        elif i in sr:
            s += 1.0 / (_RRF_K + sr[i])
        if s > 0:
            fused[i] = s
    return fused


# ---------------------------------------------------------------------------
# Embedding（异步，复用 EMBEDDING_* 配置；未配置/失败则不可用，退回纯 BM25）
# ---------------------------------------------------------------------------


class _Embedder:
    def __init__(self, settings: Any):
        self._client = None
        self._model = ""
        if settings is None:
            return
        try:
            url = getattr(settings, "EMBEDDING_URL", "") or ""
            model = getattr(settings, "EMBEDDING_MODEL", "") or ""
            if not (url and model):
                return
            base = getattr(settings, "EMBEDDING_BASE_URL", "") or url
            key = getattr(settings, "EMBEDDING_API_KEY", "") or ""
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=key, base_url=base)
            self._model = model
        except Exception as e:  # noqa: BLE001
            logger.info("[tool_disclosure] 未启用向量检索（embedding 不可用）: %s", e)
            self._client = None
            self._model = ""

    @property
    def available(self) -> bool:
        return self._client is not None and bool(self._model)

    async def aembed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.available or not texts:
            return None
        try:
            resp = await self._client.embeddings.create(input=list(texts), model=self._model)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            logger.warning("[tool_disclosure] embedding 调用失败，本次退回纯 BM25: %s", e)
            return None


# ---------------------------------------------------------------------------
# 条目 / 服务 / 混合检索器
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    name: str
    description: str
    tool: BaseTool
    type_slug: str
    service: str  # "" 表示自定义工具（无 service）
    service_description: str
    coarse_text: str
    fine_text: str  # type 标签 + 工具描述（tool 级细排文本）
    tokens: List[str] = field(default_factory=list)


@dataclass
class _Service:
    name: str
    type_slug: str
    description: str
    entries: List[_Entry]
    ds_text: str  # type + 服务描述 + 旗下各工具描述（service 级细排文本）


def _tool_leaf_name(name: str) -> str:
    """去掉 ``mcp__<server>__`` 前缀，便于可读展示。"""
    if name.startswith("mcp__") and name.count("__") >= 2:
        return name.split("__", 2)[2]
    return name


def _build_entry(tool: BaseTool) -> _Entry:
    try:
        md = dict(getattr(tool, "metadata", None) or {})
    except Exception:  # noqa: BLE001
        md = {}
    name = tool.name
    is_mcp = name.startswith("mcp__") or bool(md.get("mcp_service"))
    if is_mcp:
        service = md.get("mcp_service") or (
            name.split("__")[1] if name.startswith("mcp__") and name.count("__") >= 2 else "mcp"
        )
        type_slug = normalize_type(md.get("mcp_type"))
        service_desc = str(md.get("mcp_service_description") or "")
    else:
        service = ""
        type_slug = normalize_type(md.get("custom_type"))
        service_desc = ""
    desc = tool.description or ""
    coarse = _entry_search_text(tool)
    fine = (type_label(type_slug) + " " if type_slug else "") + desc
    return _Entry(
        name=name,
        description=desc,
        tool=tool,
        type_slug=type_slug,
        service=service,
        service_description=service_desc,
        coarse_text=coarse,
        fine_text=fine,
        tokens=_tokenize(coarse),
    )


def _build_services(mcp_entries: List[_Entry]) -> Dict[str, _Service]:
    groups: Dict[str, List[_Entry]] = {}
    for e in mcp_entries:
        groups.setdefault(e.service, []).append(e)
    services: Dict[str, _Service] = {}
    for name, ents in groups.items():
        type_slug = next((x.type_slug for x in ents if x.type_slug), "")
        desc = next((x.service_description for x in ents if x.service_description), "")
        parts: List[str] = []
        if type_slug:
            parts.append(type_label(type_slug))
        if desc:
            parts.append(desc)
        for x in ents:
            parts.append(f"{_tool_leaf_name(x.name)}: {x.description}")
        services[name] = _Service(
            name=name,
            type_slug=type_slug,
            description=desc,
            entries=ents,
            ds_text="\n".join(parts),
        )
    return services


class _Hybrid:
    """对一组条目做混合检索（BM25 + 向量 + RRF），向量不可用时退回纯 BM25。"""

    def __init__(self, entries: List[_Entry], embedder: _Embedder):
        self.entries = entries
        self.embedder = embedder
        self._n = len(entries)
        self._doc_freq: Dict[str, int] = {}
        for e in entries:
            for tok in set(e.tokens):
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1
        lengths = [len(e.tokens) for e in entries]
        self._avg_dl = (sum(lengths) / len(lengths)) if lengths else 0.0
        self._doc_vecs: Optional[List[List[float]]] = None  # 懒算并缓存

    def _bm25_scores(self, q_tokens: List[str]) -> Dict[int, float]:
        scored: Dict[int, float] = {}
        for i, e in enumerate(self.entries):
            s = _bm25_score(q_tokens, e.tokens, self._avg_dl, self._doc_freq, self._n)
            if s > 0:
                scored[i] = s
        if not scored and q_tokens:
            # 全文档共享同词时 IDF=0，用子串兜底保召回。
            for i, e in enumerate(self.entries):
                hay = (e.name + " " + e.description).lower()
                if any(t in hay for t in q_tokens):
                    scored[i] = 0.1
        return scored

    async def _doc_vectors(self) -> List[List[float]]:
        if self._doc_vecs is None:
            if self.embedder.available:
                vecs = await self.embedder.aembed([e.coarse_text for e in self.entries])
                self._doc_vecs = vecs or []
            else:
                self._doc_vecs = []
        return self._doc_vecs

    async def coarse(
        self, query: str, top_m: int
    ) -> Tuple[List[_Entry], Optional[List[float]]]:
        """返回 (粗排候选条目, query 向量)。query 向量供细排复用，避免重复编码。"""
        q_tokens = _tokenize(query)
        sparse = self._bm25_scores(q_tokens) if q_tokens else {}
        dense: Dict[int, float] = {}
        q_vec: Optional[List[float]] = None
        doc_vecs = await self._doc_vectors()
        if doc_vecs:
            qv = await self.embedder.aembed([query])
            if qv:
                q_vec = qv[0]
                for i, dv in enumerate(doc_vecs):
                    c = _cosine(q_vec, dv)
                    if c > 0:
                        dense[i] = c
        fused = _rrf_fuse(sparse, dense, self._n)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_m]
        return [self.entries[i] for i in order], q_vec


async def _rerank(
    q_vec: Optional[List[float]], texts: List[str], embedder: _Embedder
) -> Optional[List[float]]:
    """细排：对候选文本编码并与 query 向量算余弦。不可用时返回 None（保持粗排序）。"""
    if q_vec is None or not embedder.available or not texts:
        return None
    vecs = await embedder.aembed(texts)
    if not vecs or len(vecs) != len(texts):
        return None
    return [_cosine(q_vec, v) for v in vecs]


# ---------------------------------------------------------------------------
# 桥接工具参数
# ---------------------------------------------------------------------------


class _SearchArgs(BaseModel):
    query: str = Field(description="用自然语言或关键词描述你需要的能力，如 '发送邮件' 或 'github issue'。")
    limit: int = Field(default=5, description="返回的候选数量（1-20）。")


class _DescribeArgs(BaseModel):
    name: str = Field(description="要查看完整参数说明的工具名（来自 tool_search / mcp_search 结果）。")


class _InvokeArgs(BaseModel):
    name: str = Field(description="要调用的工具名（来自 tool_search / mcp_search 结果）。")
    args: Optional[dict] = Field(default=None, description="传给该工具的参数对象（键值对）。")


# ---------------------------------------------------------------------------
# 桥接工具构造
# ---------------------------------------------------------------------------


def _make_tool_search(hybrid: _Hybrid, embedder: _Embedder) -> StructuredTool:
    """自定义工具的两层披露：粗排工具 → 按 type+描述 细排 → 返回工具。"""

    async def _arun(query: str, limit: int = 5) -> str:
        limit = max(1, min(int(limit or 5), 20))
        cand, q_vec = await hybrid.coarse(query, max(20, limit * 4))
        scores = await _rerank(q_vec, [e.fine_text for e in cand], embedder)
        if scores is not None:
            cand = [cand[i] for i in sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)]
        hits = cand[:limit]
        return _ok({
            "count": len(hits),
            "tools": [
                {"name": e.name, "description": e.description, "type": e.type_slug}
                for e in hits
            ],
            "hint": "用 tool_describe(name) 看参数，再用 tool_invoke(name, args) 调用",
        })

    def _run(query: str, limit: int = 5) -> str:
        return _run_coro(_arun(query, limit))

    return StructuredTool.from_function(
        func=_run, coroutine=_arun, name=TOOL_SEARCH_NAME,
        description=(
            "在「自定义工具库」中按能力检索可用工具（混合检索+按 Type 重排，返回 name+description+type）。"
            "当必备工具无法满足、需要某个本地/自定义能力时先调用它。"
        ),
        args_schema=_SearchArgs,
    )


def _make_mcp_search(
    hybrid: _Hybrid, services: Dict[str, _Service], embedder: _Embedder
) -> StructuredTool:
    """MCP 的三层披露：Tool-as-Proxy 粗排 → 上卷服务 → 按 type+服务+工具 细排服务。"""

    async def _arun(query: str, limit: int = 5) -> str:
        limit = max(1, min(int(limit or 5), 10))
        cand_tools, q_vec = await hybrid.coarse(query, max(30, limit * 8))
        # 上卷到唯一父服务（保序）
        ordered_services: List[str] = []
        for e in cand_tools:
            if e.service and e.service not in ordered_services:
                ordered_services.append(e.service)
        cand = [services[s] for s in ordered_services if s in services]
        scores = await _rerank(q_vec, [s.ds_text for s in cand], embedder)
        if scores is not None:
            cand = [cand[i] for i in sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)]
        top = cand[:limit]
        return _ok({
            "count": len(top),
            "services": [
                {
                    "service": s.name,
                    "type": s.type_slug,
                    "description": s.description,
                    "tools": [
                        {"name": e.name, "description": e.description} for e in s.entries
                    ],
                }
                for s in top
            ],
            "hint": "选定服务后用 tool_describe(name) 看某工具参数，再用 tool_invoke(name, args) 调用",
        })

    def _run(query: str, limit: int = 5) -> str:
        return _run_coro(_arun(query, limit))

    return StructuredTool.from_function(
        func=_run, coroutine=_arun, name=MCP_SEARCH_NAME,
        description=(
            "在「MCP 服务库」中按能力检索相关服务（Tool-as-Proxy 粗排→Type/Service/Tool 重排）。"
            "返回候选服务及其工具清单；需要外部集成/MCP 能力时调用它。"
        ),
        args_schema=_SearchArgs,
    )


def _make_describe_invoke(by_name: Dict[str, _Entry]) -> List[StructuredTool]:
    def _describe(name: str) -> str:
        entry = by_name.get(name)
        if entry is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search / mcp_search 检索")
        tool = entry.tool
        return _ok({
            "name": tool.name,
            "description": tool.description or "",
            "type": entry.type_slug,
            "service": entry.service,
            "parameters": _tool_param_schema(tool),
        })

    def _invoke(name: str, args: Optional[dict] = None) -> str:
        entry = by_name.get(name)
        if entry is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search / mcp_search 检索")
        tool = entry.tool
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
        entry = by_name.get(name)
        if entry is None:
            return _err(f"未找到工具 '{name}'，请先用 tool_search / mcp_search 检索")
        try:
            result = await entry.tool.ainvoke(args or {})
            return result if isinstance(result, str) else _ok(result)
        except Exception as e:  # noqa: BLE001
            return _err(f"调用工具 '{name}' 失败: {e}")

    return [
        StructuredTool.from_function(
            func=_describe, name=TOOL_DESCRIBE_NAME,
            description="查看某个工具的完整参数 schema；在 tool_search / mcp_search 选定后、调用前使用。",
            args_schema=_DescribeArgs,
        ),
        StructuredTool.from_function(
            func=_invoke, coroutine=_ainvoke, name=TOOL_INVOKE_NAME,
            description="按名字执行额外/MCP 工具并返回结果。args 为该工具所需的参数对象。",
            args_schema=_InvokeArgs,
        ),
    ]


def _compose_overview(
    n_custom: int, n_services: int, n_mcp_tools: int, hybrid_on: bool
) -> str:
    retr = "混合检索(BM25+向量)" if hybrid_on else "BM25 检索"
    lines = ["## 额外工具库（渐进式披露 · Type→Service→Tool 分层）\n"]
    if n_custom:
        lines.append(
            f"- **自定义工具 {n_custom} 个**：`tool_search(query)` 按能力{retr} + 按 Type 重排，"
            "返回工具 name/description/type；"
        )
    if n_services:
        lines.append(
            f"- **MCP：{n_services} 个服务 / {n_mcp_tools} 个工具**：`mcp_search(query)` 先 "
            f"Tool-as-Proxy {retr} 粗排、再按 Type+Service+Tool 重排，返回候选服务及其工具清单；"
        )
    lines.append("- 选定后用 `tool_describe(name)` 看参数，再 `tool_invoke(name, args)` 执行。")
    lines.append("\n必备工具能直接满足时无需走这一层。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def build_tool_disclosure(
    extra_tools: List[BaseTool],
    context_length: Optional[int] = None,
    threshold_pct: float = 10.0,
    mode: str = "auto",
    settings: Any = None,
) -> Tuple[List[BaseTool], str]:
    """构建额外/MCP 工具的分层披露层。

    返回 ``(要绑定到模型的工具, system-prompt 概览文本)``：

    - 池子小于阈值（或 mode='off'）：直接返回这些工具本身（全部绑定），概览为空。
    - 达到阈值（或 mode='on'）：按 metadata/名称前缀自动拆分为「MCP 工具」与
      「自定义工具」两池，分别构建 `mcp_search`（三层）与 `tool_search`（两层），
      并共用 `tool_describe` / `tool_invoke`。

    ``settings`` 用于取 EMBEDDING_* 配置启用向量检索；缺省或不可用则纯 BM25。
    """
    if not extra_tools:
        return [], ""

    tokens = estimate_tokens(extra_tools)
    if not should_activate(tokens, context_length, threshold_pct, mode):
        logger.info(
            "额外工具池 %d 个（约 %d tokens）未达阈值，直接绑定。", len(extra_tools), tokens
        )
        return list(extra_tools), ""

    entries = [_build_entry(t) for t in extra_tools]
    mcp_entries = [e for e in entries if e.service]
    custom_entries = [e for e in entries if not e.service]
    by_name = {e.name: e for e in entries}

    embedder = _Embedder(settings)
    bridges: List[StructuredTool] = []

    if custom_entries:
        bridges.append(_make_tool_search(_Hybrid(custom_entries, embedder), embedder))

    services: Dict[str, _Service] = {}
    if mcp_entries:
        services = _build_services(mcp_entries)
        bridges.append(_make_mcp_search(_Hybrid(mcp_entries, embedder), services, embedder))

    bridges.extend(_make_describe_invoke(by_name))

    overview = _compose_overview(
        len(custom_entries), len(services), len(mcp_entries), embedder.available
    )
    logger.info(
        "额外工具池 %d 个（约 %d tokens）达阈值，启用分层披露："
        "自定义 %d / MCP 工具 %d（%d 服务），向量检索=%s。",
        len(entries), tokens, len(custom_entries), len(mcp_entries),
        len(services), embedder.available,
    )
    return bridges, overview
