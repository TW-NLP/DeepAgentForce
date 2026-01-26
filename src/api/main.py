"""
FastAPI 主应用
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from config.settings import settings, validate_settings
from src.api.routes import router as api_router
from src.api.websocket import setup_websocket_routes

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("=" * 70)
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 正在启动...")
    logger.info("=" * 70)

    logger.info("✅ 所有服务已就绪")
    logger.info("=" * 70)
    
    yield
    
    # 关闭时
    logger.info("👋 应用正在关闭...")


# 创建应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="支持真实状态回调和流式输出的智能搜索助手",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 REST API 路由
app.include_router(api_router, prefix="/api")

# 注册 WebSocket 路由
setup_websocket_routes(app)

# 挂载静态文件（如果存在）
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    # 首页路由
    @app.get("/")
    async def read_root():
        """返回前端页面"""
        index_file = static_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": f"欢迎使用 {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api": "/api/info"
        }
else:
    @app.get("/")
    async def read_root():
        """API 信息"""
        return {
            "message": f"欢迎使用 {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api": "/api/info"
        }


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )