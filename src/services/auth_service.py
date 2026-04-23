"""
认证服务模块
处理用户注册、登录、Token 生成和验证
多租户支持：使用 tenant_uuid 进行隔离
"""

import jwt
from datetime import datetime, timedelta, timezone
from src.models.user import User, Tenant, UserRole
from src.database.connection import SyncSessionLocal
from config.settings import get_settings
import logging
import uuid
import bcrypt
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)
settings = get_settings()


class AuthService:
    """认证服务类"""

    def __init__(self):
        self.settings = settings

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False

    def get_password_hash(self, password: str) -> str:
        """生成密码哈希"""
        try:
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"密码哈希失败: {e}")
            raise

    def create_access_token(self, user_id: int, username: str, tenant_uuid: str = None) -> str:
        """创建访问令牌 (JWT) - 使用 tenant_uuid"""
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
        payload = {
            "sub": str(user_id),
            "username": username,
            "tenant_uuid": tenant_uuid,  # 🆕 使用 tenant_uuid
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        return jwt.encode(
            payload,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )

    def create_refresh_token(self, user_id: int, tenant_uuid: str = None) -> str:
        """创建刷新令牌 - 使用 tenant_uuid"""
        expire = datetime.now(timezone.utc) + timedelta(
            days=self.settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        payload = {
            "sub": str(user_id),
            "tenant_uuid": tenant_uuid,  # 🆕 使用 tenant_uuid
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(
            payload,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )

    def verify_token(self, token: str) -> dict:
        """验证 JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token 已过期")
        except jwt.InvalidTokenError:
            raise ValueError("无效的 Token")

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = None,
    ) -> dict:
        """注册新用户 - 每个用户自动拥有独立的私有租户"""
        db = SyncSessionLocal()
        try:
            username = username.strip()
            email = email.strip().lower()

            # 验证用户名格式（基本校验）
            if len(username) < 3:
                return {"success": False, "error": "用户名至少需要3个字符"}
            if not username.replace('_', '').isalnum():
                return {"success": False, "error": "用户名只能包含字母、数字和下划线"}

            # 验证密码格式
            if len(password) < 6:
                return {"success": False, "error": "密码至少需要6个字符"}

            # 检查用户名是否已存在
            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                return {"success": False, "error": "用户名已存在，请使用其他用户名"}

            # 检查邮箱是否已存在
            existing_email = db.query(User).filter(User.email == email).first()
            if existing_email:
                return {"success": False, "error": "该邮箱已被注册，请使用其他邮箱或尝试登录"}

            # 自动为用户创建私有租户（使用用户名作为租户标识）
            tenant = Tenant(
                name=f"{username}_space",
                code=f"user_{username}",
                description=f"{username} 的个人工作空间",
            )
            db.add(tenant)
            db.flush()

            # 创建用户
            hashed_password = self.get_password_hash(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                tenant_id=tenant.id,
                tenant_uuid=tenant.uuid,
                role=UserRole.USER,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            # 生成 Token
            access_token = self.create_access_token(
                user_id=user.id,
                username=user.username,
                tenant_uuid=tenant.uuid
            )
            refresh_token = self.create_refresh_token(
                user_id=user.id,
                tenant_uuid=tenant.uuid
            )

            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "uuid": user.uuid,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "tenant_id": user.tenant_id,
                    "tenant_uuid": tenant.uuid,
                    "tenant_name": tenant.name,
                    "avatar_url": user.avatar_url,
                    "department": user.department,
                },
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
            }

        except IntegrityError as e:
            db.rollback()
            error_msg = str(e)
            logger.warning(f"注册时发生数据完整性错误: {error_msg}")

            if "users.username" in error_msg or "UNIQUE constraint failed: users.username" in error_msg:
                return {"success": False, "error": "用户名已存在，请使用其他用户名"}
            elif "users.email" in error_msg or "UNIQUE constraint failed: users.email" in error_msg:
                return {"success": False, "error": "该邮箱已被注册，请使用其他邮箱或尝试登录"}
            elif "tenants.name" in error_msg or "UNIQUE constraint failed: tenants.name" in error_msg:
                return {"success": False, "error": "租户名称已存在，请使用其他用户名重试"}
            elif "tenants.code" in error_msg or "UNIQUE constraint failed: tenants.code" in error_msg:
                return {"success": False, "error": "租户代码已存在，请使用其他用户名重试"}
            else:
                return {"success": False, "error": "注册失败，可能是数据冲突，请稍后重试"}

        except Exception as e:
            db.rollback()
            logger.error(f"注册失败: {e}")
            return {"success": False, "error": f"注册失败: {str(e)}"}
        finally:
            db.close()

    def login(self, username: str, password: str) -> dict:
        """用户登录"""
        db = SyncSessionLocal()
        try:
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()

            if not user:
                return {"success": False, "error": "用户不存在"}

            if not self.verify_password(password, user.hashed_password):
                return {"success": False, "error": "密码错误"}

            if not user.is_active:
                return {"success": False, "error": "账号已被禁用"}

            user.last_login_at = datetime.now(timezone.utc)
            db.commit()

            tenant_name = None
            tenant_uuid = user.tenant_uuid  # 🆕 获取 tenant_uuid
            if user.tenant_id:
                tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
                tenant_name = tenant.name if tenant else None

            # 生成 Token（使用 tenant_uuid）
            access_token = self.create_access_token(
                user_id=user.id,
                username=user.username,
                tenant_uuid=tenant_uuid
            )
            refresh_token = self.create_refresh_token(
                user_id=user.id,
                tenant_uuid=tenant_uuid
            )

            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "uuid": user.uuid,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "tenant_id": user.tenant_id,
                    "tenant_uuid": tenant_uuid,  # 🆕 返回 tenant_uuid
                    "tenant_name": tenant_name,
                    "avatar_url": user.avatar_url,
                    "department": user.department,
                },
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
            }

        except Exception as e:
            logger.error(f"登录失败: {e}")
            return {"success": False, "error": str(e)}
        finally:
            db.close()

    def get_current_user(self, token: str) -> dict:
        """从 Token 获取当前用户信息"""
        try:
            payload = self.verify_token(token)
            if payload.get("type") != "access":
                raise ValueError("无效的 Token 类型")

            user_id = int(payload.get("sub"))
            tenant_uuid = payload.get("tenant_uuid")  # 🆕 获取 tenant_uuid
            db = SyncSessionLocal()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    raise ValueError("用户不存在")

                if not user.is_active:
                    raise ValueError("账号已被禁用")

                return {
                    "id": user.id,
                    "uuid": user.uuid,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "tenant_id": user.tenant_id,
                    "tenant_uuid": tenant_uuid or user.tenant_uuid,  # 🆕 返回 tenant_uuid
                    "tenant_name": user.tenant.name if user.tenant else None,
                    "is_superuser": user.is_superuser,
                    "avatar_url": user.avatar_url,
                    "department": user.department,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                }
            finally:
                db.close()

        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            raise ValueError("Token 验证失败")

    def refresh_access_token(self, refresh_token: str) -> dict:
        """使用刷新令牌获取新的访问令牌"""
        try:
            payload = self.verify_token(refresh_token)
            if payload.get("type") != "refresh":
                return {"success": False, "error": "无效的 Token 类型"}

            user_id = int(payload.get("sub"))
            tenant_uuid = payload.get("tenant_uuid")  # 🆕 获取 tenant_uuid
            db = SyncSessionLocal()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return {"success": False, "error": "用户不存在"}

                if not user.is_active:
                    return {"success": False, "error": "账号已被禁用"}

                access_token = self.create_access_token(
                    user_id=user.id,
                    username=user.username,
                    tenant_uuid=tenant_uuid or user.tenant_uuid
                )

                return {
                    "success": True,
                    "access_token": access_token,
                    "token_type": "bearer",
                }
            finally:
                db.close()

        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"刷新 Token 失败: {e}")
            return {"success": False, "error": "Token 刷新失败"}


# 创建全局认证服务实例
auth_service = AuthService()
