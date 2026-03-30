"""
数据库配置和连接模块
支持 MySQL 数据库连接
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

# 获取数据库配置
settings = get_settings()

# 同步引擎（用于创建表等操作）
SYNC_DATABASE_URL = (
    f"mysql+pymysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

# 异步引擎（用于运行时操作）
ASYNC_DATABASE_URL = (
    f"mysql+aiomysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

# 创建同步引擎
sync_engine = create_engine(
    SYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# 创建异步引擎
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# 同步 Session 工厂
SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
)

# 异步 Session 工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base 类
Base = declarative_base()


def get_sync_db():
    """获取同步数据库会话"""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """获取异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_database():
    """初始化数据库 - 创建所有表"""
    try:
        Base.metadata.create_all(bind=sync_engine)
        logger.info("✅ 数据库表初始化成功")
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        raise
