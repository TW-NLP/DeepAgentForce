"""通用工具集合（参照 hermes tools 的粒度与分类思路）。

DeepAgentForce 已从 deepagents 继承文件/Shell/检索类基础工具（read_file、
write_file、edit_file、ls、glob、grep、task、write_todos、shell）以及 skills
渐进披露工具。本包在此之上补充三类**常用**工具：

- 本地实用（utils）：日期时间、计算、JSON/正则提取、文本统计、下载/保存/列举、文档抽取
- 联网检索（web）：web_search、web_fetch、http_request
- 记忆会话（memory）：memory_write、memory_search、session_search

统一通过 :func:`build_common_tools` 工厂注入租户上下文后返回 LangChain
``StructuredTool`` 列表，并附带一段用于 system prompt 的简短概览。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from langchain_core.tools import StructuredTool

from .memory import build_memory_tools
from .utils import build_util_tools
from .web import build_web_tools

__all__ = ["build_common_tools", "list_builtin_tool_meta"]

# 分类中文名（用于前端展示）
_CATEGORY_LABELS = {
    "utils": "本地实用",
    "web": "联网检索",
    "memory": "记忆会话",
}


def list_builtin_tool_meta(
    settings: Any, tenant_uuid: Optional[str]
) -> List[dict]:
    """返回内置通用工具的元信息列表：[{name, description, category, category_label}]。

    供前端「工具」页只读展示用，不参与 Agent 绑定。
    """
    out: List[dict] = []
    for category, factory in (
        ("utils", build_util_tools),
        ("web", build_web_tools),
        ("memory", build_memory_tools),
    ):
        try:
            for t in factory(settings, tenant_uuid):
                out.append({
                    "name": t.name,
                    "description": (t.description or "").strip().split("\n")[0],
                    "category": category,
                    "category_label": _CATEGORY_LABELS.get(category, category),
                })
        except Exception:  # noqa: BLE001 - 单类构建失败不影响整体列举
            continue
    return out


_OVERVIEW = """## 通用工具

除技能系统外，你还可直接调用以下常用工具：
- **本地实用**：`get_datetime` 当前时间、`calculator` 计算、`json_query` 取 JSON 值、`regex_extract` 正则提取、`text_stats` 文本统计、`download_file` 下载到产物目录、`save_text_file` 保存文本、`list_outputs` 列产物、`read_document` 抽取 pdf/docx/xlsx 文本。
- **联网检索**：`web_search` 搜索互联网、`web_fetch` 抓取网页正文、`http_request` 调用开放 REST API。
- **记忆会话**：`memory_write` 记住长期事实、`memory_search` 检索长期记忆、`session_search` 检索历史会话。

按需调用，能直接回答的简单问题无需调用工具。工具返回为结构化数据，请用自然语言总结后再回复用户。"""


def build_common_tools(
    settings: Any, tenant_uuid: Optional[str]
) -> Tuple[List[StructuredTool], str]:
    """构建通用工具：返回 (工具列表, system-prompt 概览文本)。"""
    tools: List[StructuredTool] = []
    tools.extend(build_util_tools(settings, tenant_uuid))
    tools.extend(build_web_tools(settings, tenant_uuid))
    tools.extend(build_memory_tools(settings, tenant_uuid))
    return tools, _OVERVIEW
