"""记忆与会话工具：跨对话持久化与历史检索（按租户隔离）。

- ``memory_write`` / ``memory_search``：把要长期记住的事实写入按租户隔离的
  JSON 记忆库，并支持关键词检索。
- ``session_search``：在当前租户的历史会话（HISTORY_DIR/<tenant>/session_*.json）
  中按关键词检索过往问答，便于"我们之前聊过什么"类的回溯。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _err(message: str, **extra: Any) -> str:
    payload = {"error": str(message)}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _score(text: str, terms: List[str]) -> int:
    low = text.lower()
    return sum(low.count(t) for t in terms)


def build_memory_tools(settings: Any, tenant_uuid: Optional[str]) -> List[StructuredTool]:
    tenant = tenant_uuid or "default"
    memory_dir = Path(settings.DATA_DIR) / "agent_memory"
    memory_file = memory_dir / f"{tenant}.json"
    history_dir = Path(settings.HISTORY_DIR) / tenant

    def _load_memory() -> List[dict]:
        if not memory_file.exists():
            return []
        try:
            return json.loads(memory_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_memory(items: List[dict]) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_file.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- memory_write ------------------------------------------------------
    class _WriteArgs(BaseModel):
        content: str = Field(description="要长期记住的事实/偏好/结论（一条一句最佳）。")
        tags: Optional[List[str]] = Field(default=None, description="可选标签，便于后续检索。")

    def _memory_write(content: str, tags: Optional[List[str]] = None) -> str:
        items = _load_memory()
        entry = {
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
        }
        items.append(entry)
        _save_memory(items)
        return _ok({"saved": True, "id": entry["id"], "total": len(items)})

    # ---- memory_search -----------------------------------------------------
    class _SearchArgs(BaseModel):
        query: str = Field(description="检索关键词；留空则返回全部记忆。")
        limit: int = Field(default=10, description="最多返回条数。")

    def _memory_search(query: str = "", limit: int = 10) -> str:
        items = _load_memory()
        terms = [t for t in re.split(r"\s+", query.lower()) if t]
        if terms:
            scored = [
                (it, _score(it["content"] + " " + " ".join(it.get("tags", [])), terms))
                for it in items
            ]
            hits = [it for it, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]
        else:
            hits = list(reversed(items))
        hits = hits[: max(1, limit)]
        return _ok({"count": len(hits), "memories": hits})

    # ---- session_search ----------------------------------------------------
    class _SessionArgs(BaseModel):
        query: str = Field(description="在历史会话中检索的关键词。")
        limit: int = Field(default=10, description="最多返回的匹配问答数量。")

    def _session_search(query: str, limit: int = 10) -> str:
        if not history_dir.exists():
            return _ok({"count": 0, "matches": []})
        terms = [t for t in re.split(r"\s+", query.lower()) if t]
        if not terms:
            return _err("请提供检索关键词")
        matches = []
        for session_file in history_dir.glob("session_*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            title = data.get("title", "")
            for conv in data.get("conversations", []):
                user_c = conv.get("user_content", "") or ""
                ai_c = conv.get("ai_content", "") or ""
                combined = f"{user_c}\n{ai_c}"
                s = _score(combined, terms)
                if s > 0:
                    matches.append({
                        "session_id": data.get("session_id"),
                        "title": title,
                        "timestamp": conv.get("timestamp"),
                        "user": user_c[:200],
                        "ai": ai_c[:300],
                        "_score": s,
                    })
        matches.sort(key=lambda m: m["_score"], reverse=True)
        for m in matches:
            m.pop("_score", None)
        matches = matches[: max(1, limit)]
        return _ok({"count": len(matches), "matches": matches})

    return [
        StructuredTool.from_function(
            func=_memory_write, name="memory_write",
            description="把需要跨对话长期记住的事实/用户偏好写入记忆库（按当前用户隔离）。",
            args_schema=_WriteArgs,
        ),
        StructuredTool.from_function(
            func=_memory_search, name="memory_search",
            description="按关键词检索之前写入的长期记忆；留空返回最近若干条。",
            args_schema=_SearchArgs,
        ),
        StructuredTool.from_function(
            func=_session_search, name="session_search",
            description="在当前用户的历史会话记录中按关键词检索过往问答，用于回溯之前聊过的内容。",
            args_schema=_SessionArgs,
        ),
    ]
