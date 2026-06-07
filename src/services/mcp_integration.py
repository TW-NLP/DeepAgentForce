"""MCP（Model Context Protocol）工具接入。

把外部 MCP server 暴露的工具转成 LangChain ``BaseTool``，注入到
``ConversationalAgent._collect_extra_tools()``，从而自动享受 tool_disclosure 的
渐进式披露：池子小直接绑定，超过上下文 ~10% 时走 tool_search/describe/invoke。

设计要点：
- **可选且容错**：没有配置、依赖缺失、server 连不上，都只记日志并返回 ``[]``，
  绝不影响 Agent 正常构建。MCP 是增量能力，不是必备。
- **多租户**：全局 ``data/mcp_servers.json`` + 租户 ``data/mcp_servers_<uuid>.json``
  合并（租户覆盖同名）。也支持租户 saved_config 里的 ``MCP_SERVERS`` 字段。
- **配置格式**：兼容 Claude Desktop 的 ``{"mcpServers": {...}}`` 与扁平 ``{name: conn}``。
  每个 server：有 ``command`` → stdio；有 ``url`` → streamable_http（可用 transport 覆盖为 sse）。
  支持 ``disabled``/``enabled`` 开关，字符串值支持 ``${ENV}`` 展开。
- **同步桥接**：``get_tools()`` 是协程，而 ``build_instance`` 是同步的，可能在事件
  循环内被调用，故用 ``_run_sync`` 在必要时另起线程跑，避免 "loop already running"。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.tool_taxonomy import normalize_type

logger = logging.getLogger(__name__)

# stdio / http 连接各自允许透传的字段（其余忽略，避免把无关键塞给 client）。
_STDIO_KEYS = {
    "command", "args", "env", "cwd", "encoding",
    "encoding_error_handler", "session_kwargs",
}
_HTTP_KEYS = {
    "url", "headers", "timeout", "sse_read_timeout",
    "terminate_on_close", "session_kwargs",
}
_DISABLE_KEYS = ("disabled", "enabled")


def _expand(value: Any) -> Any:
    """对字符串递归做 ${ENV} / $ENV 展开；list/dict 逐项处理。"""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _is_enabled(conn: Dict[str, Any]) -> bool:
    if conn.get("disabled") is True:
        return False
    if "enabled" in conn and conn.get("enabled") is False:
        return False
    return True


def _normalize_connection(name: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把一个 server 配置规范化为 MultiServerMCPClient 接受的 connection dict。

    返回 None 表示该 server 被禁用或配置无效（跳过，不报错）。
    """
    if not isinstance(raw, dict):
        logger.warning("[MCP] server '%s' 配置不是对象，跳过", name)
        return None
    if not _is_enabled(raw):
        logger.info("[MCP] server '%s' 已禁用，跳过", name)
        return None

    transport = raw.get("transport")

    if raw.get("command"):
        conn = {k: _expand(raw[k]) for k in _STDIO_KEYS if k in raw}
        conn.setdefault("args", [])
        conn["transport"] = "stdio"
        return conn

    if raw.get("url"):
        conn = {k: _expand(raw[k]) for k in _HTTP_KEYS if k in raw}
        # 默认 streamable_http；显式 sse / websocket 时尊重之。
        conn["transport"] = transport or "streamable_http"
        return conn

    logger.warning("[MCP] server '%s' 既无 command 也无 url，跳过", name)
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        logger.warning("[MCP] %s 顶层不是对象，忽略", path)
    except Exception as e:  # noqa: BLE001 - 配置坏掉不应影响主流程
        logger.warning("[MCP] 解析 %s 失败: %s", path, e)
    return {}


def _extract_servers(blob: Dict[str, Any]) -> Dict[str, Any]:
    """从一份配置里取出 server 映射，兼容 {'mcpServers': {...}} 与扁平形式。"""
    if not blob:
        return {}
    if isinstance(blob.get("mcpServers"), dict):
        return blob["mcpServers"]
    if isinstance(blob.get("MCP_SERVERS"), dict):
        return blob["MCP_SERVERS"]
    # 扁平：把看起来像 server 配置（dict 且含 command/url）的键收进来。
    flat = {
        k: v for k, v in blob.items()
        if isinstance(v, dict) and ("command" in v or "url" in v)
    }
    return flat


def _merge_raw_servers(settings: Any, tenant_uuid: Optional[str]) -> Dict[str, Any]:
    """汇总各来源的原始 server 配置（未规范化，保留 type/description 等附加字段）。

    合并顺序（后者覆盖同名）：
    1. ``DATA_DIR/mcp_servers.json``（全局共享）
    2. 租户 saved_config 里的 ``MCP_SERVERS`` 字段（若有）
    3. ``DATA_DIR/mcp_servers_<tenant>.json``（租户专属，优先级最高）
    """
    data_dir = Path(getattr(settings, "DATA_DIR", "data"))
    merged_raw: Dict[str, Any] = {}

    merged_raw.update(_extract_servers(_load_json(data_dir / "mcp_servers.json")))

    tenant_field = getattr(settings, "MCP_SERVERS", None)
    if isinstance(tenant_field, dict):
        merged_raw.update(_extract_servers({"mcpServers": tenant_field}))

    if tenant_uuid:
        merged_raw.update(
            _extract_servers(_load_json(data_dir / f"mcp_servers_{tenant_uuid}.json"))
        )
    return merged_raw


def load_mcp_connections(settings: Any, tenant_uuid: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """汇总并规范化所有 MCP server 连接配置（供 MCP client 使用）。"""
    connections: Dict[str, Dict[str, Any]] = {}
    for name, raw in _merge_raw_servers(settings, tenant_uuid).items():
        conn = _normalize_connection(name, raw)
        if conn is not None:
            connections[name] = conn
    return connections


def load_mcp_meta(settings: Any, tenant_uuid: Optional[str]) -> Dict[str, Dict[str, str]]:
    """每个启用 server 的 Type / 服务描述，供分层披露（mcp_search）的重排使用。

    与 :func:`load_mcp_connections` 同源合并，但保留 ``type``/``description``
    这些不进连接配置的字段。被禁用的 server 跳过。
    """
    meta: Dict[str, Dict[str, str]] = {}
    for name, raw in _merge_raw_servers(settings, tenant_uuid).items():
        if not isinstance(raw, dict) or not _is_enabled(raw):
            continue
        meta[name] = {
            "type": normalize_type(raw.get("type")),
            "description": str(raw.get("description") or "").strip(),
        }
    return meta


def _run_sync(coro):
    """同步执行协程，兼容「已有运行中的事件循环」的场景。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # 当前线程已有运行中的 loop：另起线程跑独立 loop，避免冲突。
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


async def _afetch_tools(
    connections: Dict[str, Dict[str, Any]],
    meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    meta = meta or {}
    client = MultiServerMCPClient(connections)
    # 逐 server 取，单个 server 失败不影响其余。
    tools: List[Any] = []
    for name in connections:
        try:
            server_tools = await client.get_tools(server_name=name)
            m = meta.get(name, {})
            for t in server_tools:
                # 加前缀避免不同 server 工具重名冲突，并保留可读性。
                _prefix_tool(t, name)
                # 把 service（=server 名）/ Type / 服务描述写进 metadata，
                # 供 tool_disclosure 的 mcp_search 做 Type→Service→Tool 分层重排。
                _annotate_tool(t, name, m.get("type", ""), m.get("description", ""))
            tools.extend(server_tools)
            logger.info("[MCP] server '%s' 提供 %d 个工具", name, len(server_tools))
        except Exception as e:  # noqa: BLE001
            logger.warning("[MCP] 连接 server '%s' 失败，已跳过: %s", name, e)
    return tools


def _prefix_tool(tool: Any, server: str) -> None:
    """给 MCP 工具名加 ``mcp__<server>__`` 前缀，避免跨 server 重名。"""
    try:
        original = tool.name
        if not original.startswith(f"mcp__{server}__"):
            tool.name = f"mcp__{server}__{original}"
    except Exception:  # noqa: BLE001 - 某些工具 name 只读则保持原样
        pass


def _annotate_tool(tool: Any, server: str, type_slug: str, service_description: str) -> None:
    """把 service / Type / 服务描述写进工具 metadata（分层披露重排用，容错）。"""
    try:
        md = dict(getattr(tool, "metadata", None) or {})
        md["mcp_service"] = server
        md["mcp_type"] = type_slug or ""
        md["mcp_service_description"] = service_description or ""
        tool.metadata = md
    except Exception:  # noqa: BLE001 - metadata 只读则忽略
        pass


def collect_mcp_tools(settings: Any, tenant_uuid: Optional[str]) -> List[Any]:
    """对外入口：返回当前租户可用的 MCP 工具（LangChain BaseTool 列表）。

    任何阶段出错都返回已收集到的部分（或空列表），保证不影响 Agent 构建。
    """
    try:
        connections = load_mcp_connections(settings, tenant_uuid)
    except Exception as e:  # noqa: BLE001
        logger.warning("[MCP] 加载连接配置失败: %s", e)
        return []

    if not connections:
        logger.debug("[MCP] 未配置任何 server，跳过")
        return []

    try:
        meta = load_mcp_meta(settings, tenant_uuid)
    except Exception as e:  # noqa: BLE001
        logger.warning("[MCP] 加载 server Type/描述失败，按空处理: %s", e)
        meta = {}

    logger.info("[MCP] 准备连接 %d 个 server: %s", len(connections), list(connections))
    try:
        return _run_sync(_afetch_tools(connections, meta))
    except Exception as e:  # noqa: BLE001
        logger.warning("[MCP] 获取工具失败，按无 MCP 处理: %s", e)
        return []


# ---------------------------------------------------------------------------
# 配置 CRUD（供前端「MCP 管理」页用）
# ---------------------------------------------------------------------------

_GLOBAL_FILE = "mcp_servers.json"


def _tenant_file(data_dir: Path, tenant_uuid: str) -> Path:
    return data_dir / f"mcp_servers_{tenant_uuid}.json"


def _read_servers_blob(path: Path) -> Dict[str, Any]:
    """读出文件里的 servers 映射（统一为 {name: raw} 形式）。"""
    blob = _load_json(path)
    return dict(_extract_servers(blob))


def _write_servers_blob(path: Path, servers: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class McpConfigStore:
    """管理 MCP server 配置文件（多租户）。

    - 全局 ``data/mcp_servers.json``：所有租户共享、**只读**（前端不可改）。
    - 租户 ``data/mcp_servers_<uuid>.json``：当前租户可增删改。
    同名时租户覆盖全局。
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def list_servers(self, tenant_uuid: Optional[str]) -> List[Dict[str, Any]]:
        """合并列出全局 + 租户 server，并标注来源、是否启用、传输方式。"""
        result: Dict[str, Dict[str, Any]] = {}

        for name, raw in _read_servers_blob(self.data_dir / _GLOBAL_FILE).items():
            result[name] = self._describe(name, raw, source="global", editable=False)

        if tenant_uuid:
            for name, raw in _read_servers_blob(_tenant_file(self.data_dir, tenant_uuid)).items():
                result[name] = self._describe(name, raw, source="tenant", editable=True)

        return list(result.values())

    @staticmethod
    def _describe(name: str, raw: Dict[str, Any], source: str, editable: bool) -> Dict[str, Any]:
        transport = raw.get("transport") or ("stdio" if raw.get("command") else "streamable_http")
        return {
            "name": name,
            "source": source,
            "editable": editable,
            "enabled": _is_enabled(raw),
            "transport": transport,
            "command": raw.get("command", ""),
            "args": raw.get("args", []),
            "env": raw.get("env", {}),
            "url": raw.get("url", ""),
            "headers": raw.get("headers", {}),
            # 分层披露：Type（归一化到固定类目表）+ 服务描述
            "type": normalize_type(raw.get("type")),
            "description": str(raw.get("description") or ""),
        }

    def upsert_server(
        self, tenant_uuid: str, name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """新增或更新一个租户 server。config 为原始字段（command/args/env 或 url/...）。"""
        if not tenant_uuid:
            return {"success": False, "message": "需要登录才能配置 MCP"}
        name = (name or "").strip()
        if not name or not re.match(r"^[A-Za-z0-9_\-]+$", name):
            return {"success": False, "message": "server 名称只能含字母、数字、下划线、连字符"}

        # 规范化校验：至少要能形成一个合法连接
        normalized = _normalize_connection(name, {**config, "enabled": True})
        if normalized is None:
            return {"success": False, "message": "配置无效：需要提供 command（stdio）或 url（http）"}

        path = _tenant_file(self.data_dir, tenant_uuid)
        servers = _read_servers_blob(path)
        servers[name] = self._clean_config(config)
        _write_servers_blob(path, servers)
        return {"success": True, "message": f"已保存 MCP server '{name}'", "name": name}

    @staticmethod
    def _clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """只保留有意义的字段，去掉空值。"""
        out: Dict[str, Any] = {}
        for key in ("command", "args", "env", "cwd", "url", "headers", "transport"):
            if key in config and config[key] not in (None, "", [], {}):
                out[key] = config[key]
        # 分层披露：Type 归一化到固定类目表；服务描述为自由文本。
        type_slug = normalize_type(config.get("type"))
        if type_slug:
            out["type"] = type_slug
        desc = str(config.get("description") or "").strip()
        if desc:
            out["description"] = desc
        if "disabled" in config:
            out["disabled"] = bool(config["disabled"])
        return out

    def delete_server(self, tenant_uuid: str, name: str) -> Dict[str, Any]:
        if not tenant_uuid:
            return {"success": False, "message": "需要登录才能删除 MCP"}
        path = _tenant_file(self.data_dir, tenant_uuid)
        servers = _read_servers_blob(path)
        if name not in servers:
            return {"success": False, "message": f"server '{name}' 不存在或非本租户所有"}
        servers.pop(name)
        _write_servers_blob(path, servers)
        return {"success": True, "message": f"已删除 MCP server '{name}'"}

    def toggle_server(self, tenant_uuid: str, name: str, enabled: bool) -> Dict[str, Any]:
        if not tenant_uuid:
            return {"success": False, "message": "需要登录才能修改 MCP"}
        path = _tenant_file(self.data_dir, tenant_uuid)
        servers = _read_servers_blob(path)
        if name not in servers:
            return {"success": False, "message": f"server '{name}' 不存在或非本租户所有"}
        servers[name]["disabled"] = not enabled
        servers[name].pop("enabled", None)
        _write_servers_blob(path, servers)
        state = "启用" if enabled else "禁用"
        return {"success": True, "message": f"已{state} MCP server '{name}'"}

    def test_server(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """临时连接一次某 server，返回其工具列表或错误，用于「测试连接」。"""
        conn = _normalize_connection(name or "test", {**config, "enabled": True})
        if conn is None:
            return {"success": False, "message": "配置无效：需要 command 或 url"}
        try:
            tools = _run_sync(_afetch_tools({name or "test": conn}))
            return {
                "success": True,
                "message": f"连接成功，发现 {len(tools)} 个工具",
                "tools": [
                    {"name": t.name, "description": (t.description or "")[:200]}
                    for t in tools
                ],
            }
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"连接失败：{e}"}
