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


@router.post("/logout", response_model=MessageResponse, summary="用户登出")
async def logout(authorization: Optional[str] = Header(None)):
    """
    用户登出

    - 清除本地存储的 Token
    - 服务器端 Token 将在过期后自动失效
    """
    return MessageResponse(success=True, message="登出成功")
