"""
数据库配置和连接模块
默认使用 SQLite 数据库连接
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import get_settings
import logging
import importlib.util
import os

logger = logging.getLogger(__name__)

# 获取数据库配置
settings = get_settings()

# SQLite 文件路径
SQLITE_DB_PATH = settings.SQLITE_DB_PATH
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.info("SQLite database path resolved to: %s", SQLITE_DB_PATH)
logger.info("SQLite database parent writable: %s", os.access(SQLITE_DB_PATH.parent, os.W_OK) if hasattr(os, "access") else "unknown")

# 同步引擎（用于创建表等操作）
SYNC_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# 创建同步引擎
sync_engine = create_engine(
    SYNC_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# 异步引擎（当前项目默认未使用；存在 aiosqlite 时再启用）
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{SQLITE_DB_PATH}"
if importlib.util.find_spec("aiosqlite") is not None:
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        echo=False,
    )
    AsyncSessionLocal = sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
else:
    async_engine = None
    AsyncSessionLocal = None
    logger.info("aiosqlite 未安装，跳过异步数据库引擎初始化")

# 同步 Session 工厂
SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
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
    if AsyncSessionLocal is None:
        raise RuntimeError("异步数据库会话未启用，请安装 aiosqlite 后再使用异步数据库接口")
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
