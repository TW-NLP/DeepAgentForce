import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config.settings import get_settings
from fastapi.middleware.cors import CORSMiddleware
from src.services.conversational_agent import ConversationalAgent, get_tenant_settings
from src.services.person_like_service import UserPreferenceMining
from src.services.rag import MilvusRAGPipeline
from src.services.skill_manager import SkillManager
from src.api.websocket import ConversationHistoryManager, setup_websocket_routes
from src.api.routes import router as api_router
from src.api.skills_routes import router as skills_router
from src.api.auth_routes import router as auth_router
from src.database.connection import init_database
import os

logger = logging.getLogger(__name__)

# 获取配置实例
settings = get_settings()

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
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

# 主页面路由
@app.get("/index.html")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

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
        # 🆕 多租户会话管理：key = f"{tenant_id}_{session_id}"
        self.sessions: dict[str, ConversationalAgent] = {}
        # 🆕 多租户 RAG pipeline 映射：tenant_uuid -> MilvusRAGPipeline
        self._rag_engines: dict[str, MilvusRAGPipeline] = {}

    def get_rag_engine(self, tenant_uuid: Optional[str] = None) -> MilvusRAGPipeline:
        """获取租户专属的 RAG pipeline（按需创建）"""
        key = tenant_uuid or "default"
        if key not in self._rag_engines:
            # 🆕 根据租户配置创建 RAG pipeline
            if tenant_uuid:
                tenant_settings = get_tenant_settings(tenant_uuid)
            else:
                tenant_settings = self.settings
            self._rag_engines[key] = MilvusRAGPipeline(tenant_settings)
            logger.info(f"📌 创建租户 RAG engine - tenant: {key}, EMBEDDING_URL: {tenant_settings.EMBEDDING_URL}")
        return self._rag_engines[key]

    def get_or_create_session(
        self,
        session_id: str = None,
        status_callback=None,
        tenant_uuid: str = None  # 🆕 多租户
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
        # 🆕 传递 tenant_uuid 给 ConversationalAgent
        self.sessions[session_key] = ConversationalAgent(self.settings, status_callback, tenant_uuid=tenant_uuid)
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
    print(f"⚠️ 数据库初始化失败，请检查 MySQL 配置: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )