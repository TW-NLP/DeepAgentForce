#!/usr/bin/env python3
"""
数据库迁移脚本：为 Tenant 和 User 表添加 uuid 字段
运行方式: python scripts/migrate_add_uuid.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import engine, Base, SyncSessionLocal
from src.models.user import Tenant, User, TenantApiKey
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upgrade():
    """添加 uuid 字段"""
    session = SyncSessionLocal()
    try:
        # 为现有 Tenant 添加 uuid
        tenants = session.query(Tenant).filter(Tenant.uuid == None).all()
        for tenant in tenants:
            if not tenant.uuid:
                import uuid
                tenant.uuid = str(uuid.uuid4())
                logger.info(f"为租户 {tenant.name} 生成 uuid: {tenant.uuid}")
        
        session.commit()
        logger.info("✅ Tenant 表 uuid 字段更新完成")
        
        # 为现有 User 添加 uuid
        users = session.query(User).filter(User.uuid == None).all()
        for user in users:
            if not user.uuid:
                import uuid
                user.uuid = str(uuid.uuid4())
                logger.info(f"为用户 {user.username} 生成 uuid: {user.uuid}")
        
        session.commit()
        logger.info("✅ User 表 uuid 字段更新完成")
        
        # 为现有 TenantApiKey 添加 tenant_uuid
        api_keys = session.query(TenantApiKey).filter(TenantApiKey.tenant_uuid == None).all()
        for api_key in api_keys:
            if api_key.tenant:
                api_key.tenant_uuid = api_key.tenant.uuid
                logger.info(f"为 API Key {api_key.name} 设置 tenant_uuid")
        
        session.commit()
        logger.info("✅ TenantApiKey 表 tenant_uuid 字段更新完成")
        
        logger.info("🎉 数据库迁移完成！")
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ 迁移失败: {e}")
        raise
    finally:
        session.close()


def downgrade():
    """移除 uuid 字段（不推荐）"""
    logger.warning("⚠️ 回滚操作会丢失 uuid 数据，不推荐执行")


if __name__ == "__main__":
    import uuid
    
    print("=" * 50)
    print("数据库迁移：为 Tenant 和 User 表添加 uuid 字段")
    print("=" * 50)
    
    upgrade()
