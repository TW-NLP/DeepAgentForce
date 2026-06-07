"""工具管理 API 路由。

提供两类工具的查看与（自定义工具的）增删：

- **内置通用工具**（本地实用 / 联网检索 / 记忆会话）：只读展示。
- **MCP 工具**：来自已配置 MCP server，只读展示（增删在 /mcp 路由）。
- **自定义 Python 工具**：用户上传 .py，按 tenant_uuid 隔离，可增删改查。

多租户：tenant_uuid 从 JWT Bearer token 解析（与 skills_routes 一致）。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.skills_routes import get_tenant_uuid_from_request

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 数据模型 ====================

class ToolMeta(BaseModel):
    name: str
    description: str = ""
    category: str = ""
    category_label: str = ""
    source: str = "builtin"  # builtin | custom | mcp
    type: str = ""  # 分层披露 Type（固定类目表 slug）


class CustomToolFile(BaseModel):
    tool_id: str
    source: str = "custom"
    type: str = ""  # 分层披露 Type（固定类目表 slug）
    size_bytes: int = 0
    modified_at: str = ""
    tools: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None


class ToolListResponse(BaseModel):
    success: bool
    builtin: List[ToolMeta] = Field(default_factory=list)
    mcp: List[ToolMeta] = Field(default_factory=list)
    custom: List[CustomToolFile] = Field(default_factory=list)


class SimpleResponse(BaseModel):
    success: bool
    message: str = ""
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tool_id: Optional[str] = None
    tool_names: List[str] = Field(default_factory=list)


# ==================== 辅助 ====================

def _get_settings(request: Request):
    return request.app.state.engine.settings


def _custom_manager(request: Request):
    from src.services.custom_tool_manager import CustomToolManager
    base = Path(_get_settings(request).DATA_DIR) / "agent_tools_custom"
    return CustomToolManager(base)


# ==================== API ====================

@router.get("/tools", response_model=ToolListResponse, tags=["工具管理"])
async def list_tools(request: Request):
    """列出当前租户可见的全部工具：内置通用 + MCP + 自定义。"""
    try:
        settings = _get_settings(request)
        tenant_uuid = get_tenant_uuid_from_request(request)

        # 内置通用工具
        from src.services.agent_tools import list_builtin_tool_meta
        builtin = [
            ToolMeta(source="builtin", **m)
            for m in list_builtin_tool_meta(settings, tenant_uuid)
        ]

        # MCP 工具（已配置的 server 才会有；连接失败则为空）
        mcp: List[ToolMeta] = []
        try:
            from src.services.mcp_integration import collect_mcp_tools
            for t in collect_mcp_tools(settings, tenant_uuid):
                server = t.name.split("__")[1] if t.name.startswith("mcp__") else "mcp"
                md = getattr(t, "metadata", None) or {}
                mcp.append(ToolMeta(
                    name=t.name,
                    description=(t.description or "").strip().split("\n")[0],
                    category=server,
                    category_label=f"MCP·{server}",
                    source="mcp",
                    type=md.get("mcp_type", ""),
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("列举 MCP 工具失败: %s", e)

        # 自定义工具
        custom = [CustomToolFile(**c) for c in _custom_manager(request).list_tools(tenant_uuid)]

        return ToolListResponse(success=True, builtin=builtin, mcp=mcp, custom=custom)
    except Exception as e:
        logger.error(f"列举工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/types", tags=["工具管理"])
async def list_tool_types():
    """返回固定的 Type 类目表（MCP server 与自定义工具共用，供前端下拉）。"""
    from src.services.tool_taxonomy import TOOL_TYPES
    return {"success": True, "types": TOOL_TYPES}


@router.get("/tools/template", tags=["工具管理"])
async def custom_tool_template(request: Request):
    """返回一个自定义工具的 .py 模板。"""
    return {"success": True, "code": _custom_manager(request).get_template()}


@router.get("/tools/custom/{tool_id}", tags=["工具管理"])
async def get_custom_tool(tool_id: str, request: Request):
    """获取某个自定义工具的源码。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    code = _custom_manager(request).get_tool_code(tenant_uuid, tool_id)
    if code is None:
        raise HTTPException(status_code=404, detail=f"自定义工具 '{tool_id}' 不存在")
    return {"success": True, "tool_id": tool_id, "code": code}


@router.post("/tools/validate", response_model=SimpleResponse, tags=["工具管理"])
async def validate_custom_tool(code: str = Form(...), request: Request = None):
    """静态校验自定义工具代码（不保存、不执行）。"""
    result = _custom_manager(request).validate_code(code)
    return SimpleResponse(
        success=result["valid"],
        message="校验通过" if result["valid"] else "校验失败",
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        tool_names=result.get("tool_names", []),
    )


@router.post("/tools/custom", response_model=SimpleResponse, tags=["工具管理"])
async def save_custom_tool(
    tool_id: str = Form(...),
    code: str = Form(...),
    force: bool = Form(False),
    tool_type: str = Form(""),
    request: Request = None,
):
    """新增/覆盖一个自定义 Python 工具（粘贴代码）。``tool_type`` 为固定类目表里的 Type。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能保存自定义工具")
    result = _custom_manager(request).save_tool(
        tenant_uuid, tool_id, code, force=force, tool_type=tool_type
    )
    if not result["success"]:
        return SimpleResponse(success=False, message=result["message"],
                              errors=result.get("errors", []))
    return SimpleResponse(**result)


@router.post("/tools/custom/upload", response_model=SimpleResponse, tags=["工具管理"])
async def upload_custom_tool(
    file: UploadFile = File(...),
    force: bool = Form(False),
    tool_type: str = Form(""),
    request: Request = None,
):
    """上传 .py 文件作为自定义工具（文件名即 tool_id）。``tool_type`` 为固定类目表里的 Type。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能上传自定义工具")
    filename = file.filename or "tool.py"
    if not filename.endswith(".py"):
        return SimpleResponse(success=False, message="只接受 .py 文件")
    tool_id = Path(filename).stem
    try:
        code = (await file.read()).decode("utf-8")
    except Exception as e:
        return SimpleResponse(success=False, message=f"读取文件失败：{e}")
    result = _custom_manager(request).save_tool(
        tenant_uuid, tool_id, code, force=force, tool_type=tool_type
    )
    if not result["success"]:
        return SimpleResponse(success=False, message=result["message"],
                              errors=result.get("errors", []))
    return SimpleResponse(**result)


@router.delete("/tools/custom/{tool_id}", response_model=SimpleResponse, tags=["工具管理"])
async def delete_custom_tool(tool_id: str, request: Request):
    """删除一个自定义工具。"""
    tenant_uuid = get_tenant_uuid_from_request(request)
    if tenant_uuid is None:
        raise HTTPException(status_code=401, detail="需要登录才能删除自定义工具")
    result = _custom_manager(request).delete_tool(tenant_uuid, tool_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return SimpleResponse(**result)
