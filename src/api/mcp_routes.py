"""MCP server 管理 API 路由（多租户）。

提供 MCP server 配置的查看、表单录入、删除、启停、测试连接。

存储：
- 全局 ``data/mcp_servers.json``：所有租户共享、只读。
- 租户 ``data/mcp_servers_<uuid>.json``：当前租户可增删改。

多租户：tenant_uuid 从 JWT Bearer token 解析（与 skills_routes 一致）。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.skills_routes import get_tenant_uuid_from_request

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 数据模型 ====================

class McpServerInfo(BaseModel):
    name: str
    source: str  # global | tenant
    editable: bool
    enabled: bool
    transport: str
    command: str = ""
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: Dict[str, str] = Field(default_factory=dict)
    # 分层披露：Type（固定类目表）+ 服务描述
    type: str = ""
    description: str = ""


class McpListResponse(BaseModel):
    success: bool
    servers: List[McpServerInfo] = Field(default_factory=list)


class McpServerConfig(BaseModel):
    """前端表单提交的 server 配置。"""
    name: str
    transport: str = "stdio"  # stdio | streamable_http | sse
    command: Optional[str] = ""
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = ""
    headers: Dict[str, str] = Field(default_factory=dict)
    # 分层披露：Type（固定类目表里的 slug）+ 服务描述（自由文本）
    type: str = ""
    description: str = ""


class McpToggleRequest(BaseModel):
    enabled: bool


class SimpleResponse(BaseModel):
    success: bool
    message: str = ""
    name: Optional[str] = None


class McpTestResponse(BaseModel):
    success: bool
    message: str = ""
    tools: List[Dict[str, str]] = Field(default_factory=list)


# ==================== 辅助 ====================

def _store(request: Request):
    from src.services.mcp_integration import McpConfigStore
    data_dir = Path(request.app.state.engine.settings.DATA_DIR)
    return McpConfigStore(data_dir)


def _config_payload(cfg: McpServerConfig) -> Dict[str, Any]:
    """把表单模型转为存储用的原始字段（去掉与传输无关的空字段）。"""
    payload: Dict[str, Any] = {"transport": cfg.transport}
    if cfg.transport == "stdio":
        payload["command"] = (cfg.command or "").strip()
        if cfg.args:
            payload["args"] = cfg.args
        if cfg.env:
            payload["env"] = cfg.env
    else:
        payload["url"] = (cfg.url or "").strip()
        if cfg.headers:
            payload["headers"] = cfg.headers
    # 分层披露字段（store 侧会做 Type 归一化与空值清理）
    if cfg.type:
        payload["type"] = cfg.type
    if cfg.description:
        payload["description"] = cfg.description
    return payload


# ==================== API ====================

@router.get("/mcp/servers", response_model=McpListResponse, tags=["MCP 管理"])
async def list_mcp_servers(request: Request):
    """列出当前租户可见的 MCP server（全局只读 + 租户可编辑）。"""
    try:
        tenant_uuid = get_tenant_uuid_from_request(request)
        servers = _store(request).list_servers(tenant_uuid)
        return McpListResponse(success=True, servers=[McpServerInfo(**s) for s in servers])
    except Exception as e:
        logger.error(f"列举 MCP server 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/servers", response_model=SimpleResponse, tags=["MCP 管理"])
async def upsert_mcp_server(cfg: McpServerConfig, request: Request):
    """新增/更新一个 MCP server（表单）。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能配置 MCP")
    result = _store(request).upsert_server(tenant_uuid, cfg.name, _config_payload(cfg))
    if not result["success"]:
        return SimpleResponse(success=False, message=result["message"])
    return SimpleResponse(**result)


@router.post("/mcp/servers/test", response_model=McpTestResponse, tags=["MCP 管理"])
async def test_mcp_server(cfg: McpServerConfig, request: Request):
    """测试连接：临时连一次该 server，返回工具列表或错误。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能测试 MCP")
    result = _store(request).test_server(cfg.name, _config_payload(cfg))
    return McpTestResponse(**result)


@router.post("/mcp/servers/{name}/toggle", response_model=SimpleResponse, tags=["MCP 管理"])
async def toggle_mcp_server(name: str, body: McpToggleRequest, request: Request):
    """启用/禁用一个租户 server。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能修改 MCP")
    result = _store(request).toggle_server(tenant_uuid, name, body.enabled)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return SimpleResponse(**result)


@router.delete("/mcp/servers/{name}", response_model=SimpleResponse, tags=["MCP 管理"])
async def delete_mcp_server(name: str, request: Request):
    """删除一个租户 server。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能删除 MCP")
    result = _store(request).delete_server(tenant_uuid, name)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return SimpleResponse(**result)
