"""
用户认证相关的数据模型
多租户支持：每个租户有唯一的 tenant_uuid
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.connection import Base
import enum
import uuid
from datetime import datetime


class UserRole(str, enum.Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


def generate_tenant_uuid():
    """生成唯一的租户 UUID"""
    return str(uuid.uuid4())


class Tenant(Base):
    """租户表（组织/公司）"""
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, default=generate_tenant_uuid, index=True, comment="租户唯一标识UUID")
    name = Column(String(100), unique=True, nullable=False, comment="租户名称")
    code = Column(String(50), unique=True, nullable=False, index=True, comment="租户代码")
    description = Column(Text, nullable=True, comment="租户描述")

    # 租户配置
    logo_url = Column(String(500), nullable=True, comment="Logo URL")
    contact_email = Column(String(100), nullable=True, comment="联系邮箱")
    contact_phone = Column(String(20), nullable=True, comment="联系电话")

    # 配额限制
    max_users = Column(Integer, default=10, comment="最大用户数")
    max_storage_gb = Column(Integer, default=10, comment="最大存储空间(GB)")

    # 状态
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    expired_at = Column(DateTime, nullable=True, comment="过期时间")

    # 关联
    users = relationship("User", back_populates="tenant", foreign_keys="User.tenant_id")
    api_keys = relationship("TenantApiKey", back_populates="tenant", foreign_keys="TenantApiKey.tenant_id")

    def __repr__(self):
        return f"<Tenant(id={self.id}, uuid={self.uuid}, name={self.name})>"


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, default=generate_tenant_uuid, index=True, comment="用户唯一标识UUID")
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    email = Column(String(100), unique=True, nullable=False, index=True, comment="邮箱")
    hashed_password = Column(String(255), nullable=False, comment="加密后的密码")

    # 用户信息
    full_name = Column(String(100), nullable=True, comment="真实姓名")
    phone = Column(String(20), nullable=True, comment="手机号")
    avatar_url = Column(String(500), nullable=True, comment="头像URL")
    department = Column(String(100), nullable=True, comment="部门")

    # 角色和状态
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False, comment="用户角色")
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")
    is_superuser = Column(Boolean, default=False, nullable=False, comment="是否超级管理员")

    # 租户关联（保留 tenant_id 用于数据库关联，同时用 tenant_uuid 用于路径隔离）
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True, comment="所属租户ID")
    tenant_uuid = Column(String(36), ForeignKey("tenants.uuid"), nullable=True, index=True, comment="所属租户UUID")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    last_login_at = Column(DateTime, nullable=True, comment="最后登录时间")

    # Token 相关
    refresh_token = Column(String(500), nullable=True, comment="刷新令牌")
    token_expire_at = Column(DateTime, nullable=True, comment="令牌过期时间")

    # 关联
    tenant = relationship("Tenant", back_populates="users", foreign_keys="User.tenant_id")

    def __repr__(self):
        return f"<User(id={self.id}, uuid={self.uuid}, username={self.username})>"


class TenantApiKey(Base):
    """租户 API Key 表"""
    __tablename__ = "tenant_api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True, comment="所属租户ID")
    tenant_uuid = Column(String(36), ForeignKey("tenants.uuid"), nullable=True, index=True, comment="所属租户UUID")
    name = Column(String(100), nullable=False, comment="Key 名称")
    api_key = Column(String(100), unique=True, nullable=False, index=True, comment="API Key")
    secret_key = Column(String(255), nullable=False, comment="Secret Key (加密存储)")

    # 使用限制
    max_requests_per_day = Column(Integer, default=1000, comment="每日最大请求数")
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")

    # 使用统计
    used_requests_today = Column(Integer, default=0, comment="今日已用请求数")
    last_used_at = Column(DateTime, nullable=True, comment="最后使用时间")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    expired_at = Column(DateTime, nullable=True, comment="过期时间")

    # 关联
    tenant = relationship("Tenant", back_populates="api_keys", foreign_keys="TenantApiKey.tenant_id")

    def __repr__(self):
        return f"<TenantApiKey(id={self.id}, name={self.name}, tenant_uuid={self.tenant_uuid})>"
