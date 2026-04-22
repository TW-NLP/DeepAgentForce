"""
认证 API 路由
处理用户注册、登录、登出等请求
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from src.services.auth_service import auth_service
from src.database.connection import SyncSessionLocal
from src.models.user import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["认证"])

# 安全方案
security = HTTPBearer(auto_error=False)


# ==================== 请求/响应模型 ====================

class RegisterRequest(BaseModel):
    """注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    full_name: Optional[str] = Field(None, max_length=100, description="真实姓名")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")


class TokenRefreshRequest(BaseModel):
    """刷新 Token 请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class UserUpdateRequest(BaseModel):
    """用户资料更新请求"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="用户名")
    email: Optional[EmailStr] = Field(None, description="邮箱")
    full_name: Optional[str] = Field(None, max_length=100, description="真实姓名")
    avatar_url: Optional[str] = Field(None, max_length=500, description="头像URL")


class UserResponse(BaseModel):
    """用户信息响应"""
    id: int
    uuid: str = ""
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    tenant_id: Optional[int] = None
    tenant_uuid: Optional[str] = None  # 🆕 新增
    tenant_name: Optional[str] = None
    avatar_url: Optional[str] = None
    department: Optional[str] = None
    is_superuser: Optional[bool] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AuthResponse(BaseModel):
    """认证响应"""
    success: bool
    user: Optional[UserResponse] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = "bearer"
    error: Optional[str] = None


class MessageResponse(BaseModel):
    """通用消息响应"""
    success: bool
    message: str


# ==================== API 路由 ====================

@router.post("/register", response_model=AuthResponse, summary="用户注册")
async def register(request: RegisterRequest):
    """
    注册新用户

    - 每个用户自动拥有独立的私有工作空间
    - 注册成功后自动登录并返回 Token
    """
    logger.info(f"收到注册请求: {request.username}, {request.email}")

    result = auth_service.register_user(
        username=request.username,
        email=request.email,
        password=request.password,
        full_name=request.full_name,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    return AuthResponse(**result)


@router.post("/login", response_model=AuthResponse, summary="用户登录")
async def login(request: LoginRequest):
    """
    用户登录

    - 支持用户名或邮箱登录
    - 返回访问令牌和刷新令牌
    """
    logger.info(f"收到登录请求: {request.username}")

    result = auth_service.login(
        username=request.username,
        password=request.password,
    )

    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error"))

    return AuthResponse(**result)


@router.post("/refresh", response_model=AuthResponse, summary="刷新 Token")
async def refresh_token(request: TokenRefreshRequest):
    """
    使用刷新令牌获取新的访问令牌
    """
    result = auth_service.refresh_access_token(request.refresh_token)

    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error"))

    return AuthResponse(**result)


@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_me(authorization: Optional[str] = Header(None)):
    """
    获取当前登录用户的信息

    需要在请求头中携带: Authorization: Bearer <access_token>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    token = authorization.replace("Bearer ", "")

    try:
        user = auth_service.get_current_user(token)
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.patch("/me", response_model=UserResponse, summary="更新当前用户信息")
async def update_me(request: UserUpdateRequest, authorization: Optional[str] = Header(None)):
    """
    更新当前登录用户的资料
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    token = authorization.replace("Bearer ", "")
    try:
        payload = auth_service.verify_token(token)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="无效的 Token 类型")

        user_id = int(payload.get("sub"))
        db = SyncSessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="用户不存在")
            if not user.is_active:
                raise HTTPException(status_code=403, detail="账号已被禁用")

            if request.username is not None:
                normalized_username = request.username.strip()
                if len(normalized_username) < 3:
                    raise HTTPException(status_code=400, detail="用户名至少需要3个字符")
                if not normalized_username.replace('_', '').isalnum():
                    raise HTTPException(status_code=400, detail="用户名只能包含字母、数字和下划线")
                existing_user = db.query(User).filter(User.username == normalized_username, User.id != user.id).first()
                if existing_user:
                    raise HTTPException(status_code=400, detail="用户名已存在，请使用其他用户名")
                user.username = normalized_username

            if request.email is not None:
                normalized_email = request.email.strip().lower()
                existing_email = db.query(User).filter(User.email == normalized_email, User.id != user.id).first()
                if existing_email:
                    raise HTTPException(status_code=400, detail="该邮箱已被其他用户使用")
                user.email = normalized_email

            if request.full_name is not None:
                user.full_name = request.full_name.strip() or None

            if request.avatar_url is not None:
                user.avatar_url = request.avatar_url.strip() or None

            db.commit()
            db.refresh(user)

            tenant_name = user.tenant.name if user.tenant else None
            return UserResponse(
                id=user.id,
                uuid=user.uuid,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                tenant_id=user.tenant_id,
                tenant_uuid=user.tenant_uuid,
                tenant_name=tenant_name,
                avatar_url=user.avatar_url,
                department=user.department,
                is_superuser=user.is_superuser,
                created_at=user.created_at.isoformat() if user.created_at else None,
                updated_at=user.updated_at.isoformat() if user.updated_at else None,
            )
        finally:
            db.close()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"更新用户信息失败: {e}")
        raise HTTPException(status_code=500, detail="更新用户信息失败")


@router.post("/logout", response_model=MessageResponse, summary="用户登出")
async def logout(authorization: Optional[str] = Header(None)):
    """
    用户登出

    - 清除本地存储的 Token
    - 服务器端 Token 将在过期后自动失效
    """
    return MessageResponse(success=True, message="登出成功")
