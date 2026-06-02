"""联网检索工具：复用项目已有的 TAVILY / FIRECRAWL 密钥。

- ``web_search``：Tavily 搜索，返回结构化结果（title/url/snippet）。
- ``web_fetch``：抓取单个网页正文，有 Firecrawl 密钥优先用其 markdown，否则
  退化为 requests + 简单 HTML 去标签。
- ``http_request``：通用 REST 调用（GET/POST 等），用于访问开放 API。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 12000


def _err(message: str, **extra: Any) -> str:
    payload = {"error": str(message)}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _truncate(text: str) -> str:
    if len(text) > _MAX_BODY_CHARS:
        return text[:_MAX_BODY_CHARS] + f"\n…（已截断，共 {len(text)} 字符）"
    return text


def build_web_tools(settings: Any, tenant_uuid: Optional[str]) -> List[StructuredTool]:
    tavily_key = getattr(settings, "TAVILY_API_KEY", "") or ""
    firecrawl_key = getattr(settings, "FIRECRAWL_API_KEY", "") or ""

    # ---- web_search --------------------------------------------------------
    class _SearchArgs(BaseModel):
        query: str = Field(description="搜索关键词。")
        max_results: int = Field(default=5, description="返回结果数量（建议 3-10）。")

    def _web_search(query: str, max_results: int = 5) -> str:
        if not tavily_key:
            return _err("未配置 TAVILY_API_KEY，无法联网搜索")
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            resp = client.search(query=query, max_results=max(1, min(max_results, 10)))
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("content", "") or "")[:500],
                }
                for r in resp.get("results", [])
            ]
            return _ok({"query": query, "count": len(results), "results": results})
        except Exception as e:
            return _err(f"搜索失败: {e}")

    # ---- web_fetch ---------------------------------------------------------
    class _FetchArgs(BaseModel):
        url: str = Field(description="要抓取正文的网页 http/https 地址。")

    def _web_fetch(url: str) -> str:
        if not url.lower().startswith(("http://", "https://")):
            return _err("仅支持 http/https 链接")
        if firecrawl_key:
            try:
                from firecrawl import FirecrawlApp
                app = FirecrawlApp(api_key=firecrawl_key)
                data = app.scrape_url(url, params={"formats": ["markdown"]})
                content = ""
                if isinstance(data, dict):
                    content = data.get("markdown") or data.get("content") or ""
                else:
                    content = getattr(data, "markdown", "") or ""
                if content:
                    return _ok({"url": url, "content": _truncate(content)})
            except Exception as e:
                logger.warning("Firecrawl 抓取失败，回退 requests: %s", e)
        try:
            import requests
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return _ok({"url": url, "content": _truncate(_strip_html(r.text))})
        except Exception as e:
            return _err(f"抓取失败: {e}")

    # ---- http_request ------------------------------------------------------
    class _HttpArgs(BaseModel):
        url: str = Field(description="目标 http/https 地址。")
        method: str = Field(default="GET", description="HTTP 方法：GET/POST/PUT/DELETE 等。")
        headers: Optional[dict] = Field(default=None, description="可选请求头。")
        params: Optional[dict] = Field(default=None, description="可选 URL 查询参数。")
        json_body: Optional[dict] = Field(default=None, description="可选 JSON 请求体。")

    def _http_request(
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> str:
        if not url.lower().startswith(("http://", "https://")):
            return _err("仅支持 http/https 链接")
        try:
            import requests
            resp = requests.request(
                method=method.upper(), url=url, headers=headers,
                params=params, json=json_body, timeout=30,
            )
            ctype = resp.headers.get("Content-Type", "")
            body: Any
            if "application/json" in ctype:
                try:
                    body = resp.json()
                    body_text = json.dumps(body, ensure_ascii=False)
                except ValueError:
                    body_text = resp.text
            else:
                body_text = resp.text
            return _ok({
                "status": resp.status_code,
                "content_type": ctype,
                "body": _truncate(body_text),
            })
        except Exception as e:
            return _err(f"请求失败: {e}")

    return [
        StructuredTool.from_function(
            func=_web_search, name="web_search",
            description="用 Tavily 搜索互联网，返回标题、链接与摘要列表。适合查实时信息、新闻、资料。",
            args_schema=_SearchArgs,
        ),
        StructuredTool.from_function(
            func=_web_fetch, name="web_fetch",
            description="抓取指定网页的正文内容（优先 Firecrawl markdown，否则去 HTML 标签）。",
            args_schema=_FetchArgs,
        ),
        StructuredTool.from_function(
            func=_http_request, name="http_request",
            description="发起通用 HTTP 请求（GET/POST/...），用于调用开放 REST API，返回状态码与响应体。",
            args_schema=_HttpArgs,
        ),
    ]
