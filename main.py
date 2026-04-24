import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config.settings import get_settings
from fastapi.middleware.cors import CORSMiddleware
from src.services.conversational_agent import ConversationalAgent, get_tenant_settings
from src.services.person_like_service import UserPreferenceMining
from src.services.rag import ChromaRAGPipeline
from src.services.skill_manager import SkillManager
from src.services.proofread_service import ProofreadService
from src.api.websocket import ConversationHistoryManager, setup_websocket_routes
from src.api.routes import router as api_router
from src.api.skills_routes import router as skills_router
from src.api.auth_routes import router as auth_router
from src.database.connection import init_database
import os
import sys
import time
from pathlib import Path
import threading
import urllib.request

logger = logging.getLogger(__name__)

# 获取配置实例
settings = get_settings()

def _resource_root() -> Path:
    """获取静态资源根目录，兼容打包运行模式"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return Path(__file__).resolve().parent

RESOURCE_ROOT = _resource_root()
_BROWSER_OPENED = False

app = FastAPI()

# 添加 CORS 中间件 - 使用配置中的 CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
STATIC_DIR = os.path.join(str(RESOURCE_ROOT), "static")
IMAGES_DIR = os.path.join(str(RESOURCE_ROOT), "images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# 首页路由 - 返回 index.html
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Skills 页面路由
@app.get("/skills.html")
async def skills():
    return FileResponse(os.path.join(STATIC_DIR, "skills.html"))

# 登录页面路由
@app.get("/login.html")
async def login():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))

# 注册页面路由
@app.get("/register.html")
async def register():
    return FileResponse(os.path.join(STATIC_DIR, "register.html"))

# 帮助手册页面路由
@app.get("/help.html")
async def help_page():
    return FileResponse(os.path.join(STATIC_DIR, "help.html"))

# 主页面路由
@app.get("/index.html")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _auto_open_browser() -> None:
    """在桌面打包模式下自动打开首页，避免双击 app 后看起来没有反应。"""
    global _BROWSER_OPENED

    if _BROWSER_OPENED:
        return
    if not getattr(sys, "frozen", False):
        return

    auto_open = os.getenv("AUTO_OPEN_BROWSER", "true").strip().lower()
    if auto_open in {"0", "false", "no", "off"}:
        return

    url = f"http://127.0.0.1:{settings.PORT}/"
    _BROWSER_OPENED = True

    def _open() -> None:
        try:
            import webbrowser

            webbrowser.open(url)
            logger.info("Opened browser for packaged app: %s", url)
        except Exception as exc:
            logger.warning("Failed to open browser automatically: %s", exc)

    threading.Timer(1.0, _open).start()


def _wait_for_http_server(url: str, timeout: float = 30.0) -> None:
    """等待本地服务启动完成，再显示桌面窗口。"""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)

    raise RuntimeError(f"Timed out waiting for local server at {url}") from last_error


def _launch_desktop_window(url: str) -> None:
    """在打包后的桌面应用中打开原生窗口。"""
    try:
        import webview
    except Exception as exc:
        raise RuntimeError(
            "pywebview is required for the packaged desktop app"
        ) from exc

    window = webview.create_window(
        "DeepAgentForce",
        url=url,
        width=1440,
        height=960,
        min_size=(1200, 800),
        background_color="#ffffff",
    )
    logger.info("Launching native desktop window: %s", url)

    def _maximize_window() -> None:
        try:
            window.maximize()
        except Exception as exc:
            logger.debug("Failed to maximize desktop window: %s", exc)

    webview.start(_maximize_window, debug=settings.DEBUG)


@app.on_event("startup")
async def _on_startup() -> None:
    if not getattr(sys, "frozen", False):
        _auto_open_browser()

class DeepAgentForce:
    def __init__(self):
        self.settings = get_settings()
        self.user_preference = UserPreferenceMining(self.settings)
        self.history_manager = ConversationHistoryManager(self.settings.HISTORY_DIR)
        # 🆕 多租户 SkillManager：内置 Skills + 用户自定义 Skills
        self.skill_manager = SkillManager(
            builtin_skills_dir=self.settings.SKILL_DIR,
            user_skills_base_dir=self.settings.USER_SKILL_DIR
        )
        # 🆕 校对服务
        self.proofread_service = ProofreadService(self.settings)
        # 🆕 多租户会话管理：key = f"{tenant_id}_{session_id}"
        self.sessions: dict[str, ConversationalAgent] = {}
        # 🆕 多租户 RAG pipeline 映射：tenant_uuid -> ChromaRAGPipeline
        self._rag_engines: dict[str, ChromaRAGPipeline] = {}

    def get_rag_engine(self, tenant_uuid: Optional[str] = None) -> ChromaRAGPipeline:
        """获取租户专属的 RAG pipeline（按需创建）"""
        key = tenant_uuid or "default"
        if key not in self._rag_engines:
            # 🆕 根据租户配置创建 RAG pipeline
            if tenant_uuid:
                tenant_settings = get_tenant_settings(tenant_uuid)
            else:
                tenant_settings = self.settings
            self._rag_engines[key] = ChromaRAGPipeline(tenant_settings)
            logger.info(f"📌 创建租户 RAG engine - tenant: {key}, EMBEDDING_URL: {tenant_settings.EMBEDDING_URL}")
        return self._rag_engines[key]

    def get_or_create_session(
        self,
        session_id: str = None,
        status_callback=None,
        tenant_uuid: str = None,  # 🆕 多租户
    ) -> tuple[str, ConversationalAgent]:
        # 🆕 多租户会话 key
        session_key = f"{tenant_uuid}_{session_id}" if tenant_uuid else session_id

        if session_key and session_key in self.sessions:
            if status_callback:
                self.sessions[session_key].status_callback = status_callback
            return session_id, self.sessions[session_key]

        import uuid
        sid = session_id or str(uuid.uuid4())
        session_key = f"{tenant_uuid}_{sid}" if tenant_uuid else sid
        # 🆕 传递 tenant_uuid 给 ConversationalAgent（用于 Skills 隔离）
        self.sessions[session_key] = ConversationalAgent(
            self.settings,
            status_callback,
            tenant_uuid=tenant_uuid,
        )
        return sid, self.sessions[session_key]

    def init_service(self, tenant_uuid: Optional[str] = None):
        # 🆕 多租户支持：从租户配置文件加载设置
        if tenant_uuid:
            self.settings = get_tenant_settings(tenant_uuid)
        else:
            self.settings = get_settings()
        self.user_preference = UserPreferenceMining(self.settings)
        self.history_manager = ConversationHistoryManager(self.settings.HISTORY_DIR)
        # 🆕 多租户 SkillManager：内置 Skills + 用户自定义 Skills
        self.skill_manager = SkillManager(
            builtin_skills_dir=self.settings.SKILL_DIR,
            user_skills_base_dir=self.settings.USER_SKILL_DIR
        )
        # 🆕 重新初始化校对服务（使用新配置）
        self.proofread_service = ProofreadService(self.settings)
        # 🆕 重置多租户会话和 RAG engines
        self.sessions = {}
        self._rag_engines = {}
        
    

# --- 关键启动步骤 ---
engine = DeepAgentForce()
app.state.engine = engine  # 存入全局状态

app.include_router(api_router, prefix="/api")  # 挂载基础 API 路由
app.include_router(skills_router, prefix="/api")  # 挂载 Skill 管理路由
app.include_router(auth_router, prefix="/api")   # 挂载认证路由
setup_websocket_routes(app) # 挂载 WebSocket

# 初始化数据库
try:
    init_database()
except Exception as e:
    print(f"⚠️ 数据库初始化失败，请检查 SQLite 配置: {e}")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description='DeepAgentForce')
    parser.add_argument('--host', default=settings.HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=settings.PORT, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=settings.DEBUG, help='Enable auto-reload')
    args = parser.parse_args()

    if getattr(sys, "frozen", False):
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            reload=False,
            log_level="info",
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        local_url = f"http://127.0.0.1:{args.port}/"
        _wait_for_http_server(local_url)
        _launch_desktop_window(local_url)
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
