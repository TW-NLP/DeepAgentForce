"""
Skill 管理 API 路由
提供 Skill 的增删改查、导入导出等接口

多租户支持：
- 内置 Skills：所有用户可见
- 用户自定义 Skills：按 tenant_uuid 隔离
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
from src.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 多租户辅助函数 ====================

def get_tenant_uuid_from_request(request: Request) -> Optional[str]:
    """
    从请求中提取 tenant_uuid
    用于多租户 Skills 隔离
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.replace("Bearer ", "")
    try:
        payload = auth_service.verify_token(token)
        return payload.get("tenant_uuid")
    except Exception:
        return None


# ==================== 数据模型 ====================

class SkillInfo(BaseModel):
    """Skill 基本信息"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = Field(default_factory=list)
    path: str
    scripts: List[Dict[str, str]] = Field(default_factory=list)
    script_count: int = 0
    script_names: List[str] = Field(default_factory=list)
    size_bytes: int = 0
    file_count: int = 0
    summary: str = ""
    is_builtin: bool = False
    is_custom: bool = False
    created_at: str = ""
    modified_at: str = ""


class SkillListResponse(BaseModel):
    """Skill 列表响应"""
    success: bool
    total: int
    skills: List[SkillInfo]


class SkillDetailResponse(BaseModel):
    """Skill 详情响应"""
    success: bool
    skill: Optional[SkillInfo] = None


class SkillContentResponse(BaseModel):
    """Skill 内容响应"""
    success: bool
    skill_md: Optional[str] = None
    scripts: Optional[Dict[str, str]] = None


class SkillInstallRequest(BaseModel):
    """Skill 安装请求"""
    skill_name: str
    skill_md: str
    scripts: Dict[str, str]
    force: bool = False


class SkillInstallResponse(BaseModel):
    """Skill 安装响应"""
    success: bool
    message: str
    skill_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SkillDeleteResponse(BaseModel):
    """Skill 删除响应"""
    success: bool
    message: str


class SkillExportResponse(BaseModel):
    """Skill 导出响应"""
    success: bool
    skill_md: Optional[str] = None
    scripts: Optional[Dict[str, str]] = None
    exported_at: Optional[str] = None


class SkillValidationResponse(BaseModel):
    """Skill 验证响应"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SkillTemplateResponse(BaseModel):
    """Skill 模板响应"""
    success: bool
    skill_md: str
    scripts: Dict[str, str]


# ==================== API 实现 ====================

def get_skill_manager(request: Request) -> Any:
    """从应用状态获取 SkillManager"""
    return request.app.state.engine.skill_manager


@router.get("/skills", response_model=SkillListResponse, tags=["Skill 管理"])
async def list_skills(request: Request):
    """
    获取当前用户的 Skills 列表

    返回合并结果：
    - 内置 Skills（所有用户可见）
    - 用户自己的 Skills（按 tenant_uuid 隔离）
    """
    try:
        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕 获取 tenant_uuid
        skills = skill_manager.list_skills(tenant_uuid=tenant_uuid)  # 🆕 传递 tenant_uuid

        return SkillListResponse(
            success=True,
            total=len(skills),
            skills=skills
        )
    except Exception as e:
        logger.error(f"获取 Skill 列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/{skill_id}", response_model=SkillDetailResponse, tags=["Skill 管理"])
async def get_skill(skill_id: str, request: Request):
    """
    获取指定 Skill 的详细信息
    🆕 多租户：用户只能访问内置 Skills 和自己创建的 Skills
    """
    try:
        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕
        skill = skill_manager.get_skill(skill_id, tenant_uuid=tenant_uuid)  # 🆕

        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' 不存在或无权限访问")

        return SkillDetailResponse(
            success=True,
            skill=skill
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 Skill 详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/{skill_id}/content", response_model=SkillContentResponse, tags=["Skill 管理"])
async def get_skill_content(skill_id: str, request: Request):
    """
    获取 Skill 的完整内容 (SKILL.md + scripts)
    🆕 多租户：验证用户是否有权限访问
    """
    try:
        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕
        content = skill_manager.get_skill_content(skill_id, tenant_uuid=tenant_uuid)  # 🆕

        if not content:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' 不存在或无权限访问")

        # 提取内容
        skill_md = content.get('SKILL.md', '')
        scripts = content.get('scripts', {})

        return SkillContentResponse(
            success=True,
            skill_md=skill_md,
            scripts=scripts
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 Skill 内容失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skills/validate", response_model=SkillValidationResponse, tags=["Skill 管理"])
async def validate_skill(
    skill_md: str = Form(...),
    scripts: str = Form(...),
    request: Request = None
):
    """
    验证 Skill 规范的合法性
    """
    try:
        # 解析 scripts
        scripts_dict = json.loads(scripts) if scripts else {}

        # 使用共享的配置
        from src.services.skill_manager import SkillManager
        from config.settings import get_settings

        settings = get_settings()
        skill_manager = SkillManager(settings.SKILL_DIR, settings.USER_SKILL_DIR)

        validation = skill_manager.validate_skill(skill_md, scripts_dict)

        return SkillValidationResponse(**validation)
    except json.JSONDecodeError as e:
        return SkillValidationResponse(
            valid=False,
            errors=[f"scripts JSON 解析失败: {str(e)}"],
            warnings=[]
        )
    except Exception as e:
        logger.error(f"验证 Skill 失败: {e}")
        return SkillValidationResponse(
            valid=False,
            errors=[str(e)],
            warnings=[]
        )


@router.post("/skills/install", response_model=SkillInstallResponse, tags=["Skill 管理"])
async def install_skill(
    skill_name: str = Form(...),
    skill_md: str = Form(...),
    scripts: str = Form(...),
    force: bool = Form(False),
    request: Request = None
):
    """
    安装新的 Skill (从前端上传)
    🆕 多租户：用户创建的 Skills 存储到用户专属目录
    """
    try:
        # 解析 scripts
        scripts_dict = json.loads(scripts) if scripts else {}

        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕 获取 tenant_uuid

        if tenant_uuid is None:
            raise HTTPException(status_code=401, detail="需要登录才能安装 Skill")

        result = skill_manager.install_skill(
            skill_name=skill_name,
            skill_md_content=skill_md,
            scripts=scripts_dict,
            tenant_uuid=tenant_uuid,  # 🆕 传递 tenant_uuid
            force=force
        )

        return SkillInstallResponse(**result)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        return SkillInstallResponse(
            success=False,
            message=f"scripts JSON 解析失败: {str(e)}",
            errors=[str(e)]
        )
    except Exception as e:
        logger.error(f"安装 Skill 失败: {e}")
        return SkillInstallResponse(
            success=False,
            message=str(e),
            errors=[str(e)]
        )


@router.delete("/skills/{skill_id}", response_model=SkillDeleteResponse, tags=["Skill 管理"])
async def uninstall_skill(skill_id: str, request: Request):
    """
    卸载指定的 Skill
    🆕 多租户：用户只能删除自己创建的 Skills
    """
    try:
        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕
        result = skill_manager.uninstall_skill(skill_id, tenant_uuid=tenant_uuid)  # 🆕

        if not result['success']:
            if "内置" in result['message'] or "Builtin" in result['message']:
                raise HTTPException(status_code=403, detail=result['message'])
            if "无权限" in result['message'] or "not exist" in result['message'].lower():
                raise HTTPException(status_code=404, detail=result['message'])
            raise HTTPException(status_code=404, detail=result['message'])

        return SkillDeleteResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"卸载 Skill 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/{skill_id}/export", response_model=SkillExportResponse, tags=["Skill 管理"])
async def export_skill(skill_id: str, request: Request):
    """
    导出 Skill 为可分享的格式
    🆕 多租户：用户只能导出内置 Skills 和自己创建的 Skills
    """
    try:
        skill_manager = get_skill_manager(request)
        tenant_uuid = get_tenant_uuid_from_request(request)  # 🆕
        export_data = skill_manager.export_skill(skill_id, tenant_uuid=tenant_uuid)  # 🆕

        if not export_data:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' 不存在或无权限访问")

        return SkillExportResponse(
            success=True,
            **export_data
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出 Skill 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/template", response_model=SkillTemplateResponse, tags=["Skill 管理"])
async def get_skill_template(request: Request):
    """
    获取 Skill 开发模板
    """
    try:
        skill_manager = get_skill_manager(request)
        template = skill_manager.get_skill_template()
        
        return SkillTemplateResponse(
            success=True,
            **template
        )
    except Exception as e:
        logger.error(f"获取模板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
