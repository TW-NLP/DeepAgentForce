"""工具 / MCP 服务的固定 Type 类目表（单一真源）。

参照 Hi-RAG（A Hierarchical Framework for Scalable and Generalizable Tool
Selection）论文 MCPBench 的 8 个 domain。Type 是 ``Type → Service → Tool`` 分层
披露里最粗的一层：MCP server 与自定义工具在添加时都从这张表里选一个 Type，
用于 :mod:`src.services.tool_disclosure` 的细粒度重排（fine-grained re-ranking）。

设计要点：
- **固定集合**：前端下拉、配置存储、检索重排都引用同一份 :data:`TOOL_TYPES`，
  避免前后端两处维护；前端经 ``GET /api/tools/types`` 拉取。
- **严格论文口径**：只有这 8 类，不设 ``other`` 兜底。
- **向后兼容**：历史数据没有 Type 时按「空 Type」处理（:func:`normalize_type`
  返回 ``""``），仍可被检索，只是少了 Type 这一路信号，不报错。
"""

from __future__ import annotations

from typing import Dict, List

# 顺序即前端下拉展示顺序。slug 为存储/检索用的稳定标识，label 为中文展示名。
TOOL_TYPES: List[Dict[str, str]] = [
    {"slug": "file-systems", "label": "文件系统"},
    {"slug": "search", "label": "搜索"},
    {"slug": "research", "label": "研究"},
    {"slug": "location", "label": "位置"},
    {"slug": "media", "label": "多媒体"},
    {"slug": "calendar", "label": "日历"},
    {"slug": "browser", "label": "浏览器"},
    {"slug": "finance", "label": "金融"},
]

# slug -> label 的快速索引，以及合法 slug 集合。
_LABEL_BY_SLUG: Dict[str, str] = {t["slug"]: t["label"] for t in TOOL_TYPES}
TYPE_SLUGS = frozenset(_LABEL_BY_SLUG.keys())


def normalize_type(value: object) -> str:
    """把任意输入归一化为合法 Type slug；非法 / 缺失返回 ``""``（空 Type）。

    容错地接受大小写、首尾空白、以及中文 label（反查 slug），方便前端或历史
    数据直接传入。
    """
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in TYPE_SLUGS:
        return lowered
    # 允许传中文 label，反查 slug。
    for slug, label in _LABEL_BY_SLUG.items():
        if text == label:
            return slug
    return ""


def is_valid_type(value: object) -> bool:
    """是否为合法 Type（空 Type 视为不合法，用于「必须选」的校验场景）。"""
    return normalize_type(value) != ""


def type_label(slug: object) -> str:
    """返回 slug 对应的中文展示名；未知 slug 原样返回（空则给占位）。"""
    s = normalize_type(slug)
    if s:
        return _LABEL_BY_SLUG[s]
    raw = str(slug or "").strip()
    return raw or "未分类"
