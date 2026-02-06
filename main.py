import logging
from fastapi import FastAPI
from config.settings import Settings
from fastapi.middleware.cors import CORSMiddleware
from src.services.conversational_agent import ConversationalAgent
from src.services.person_like_service import UserPreferenceMining
from src.services.rag import MilvusRAGPipeline
from src.api.websocket import ConversationHistoryManager, setup_websocket_routes
from src.api.routes import router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境方便），生产环境建议改为前端具体地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, etc.)
    allow_headers=["*"],  # 允许所有 Header
)

class DeepAgentForce:
    def __init__(self):
        self.settings = Settings()
        self.user_preference = UserPreferenceMining(self.settings)
        self.rag_engine = MilvusRAGPipeline(self.settings)
        # 历史管理器
        self.history_manager = ConversationHistoryManager(self.settings.HISTORY_DIR)        
        self.sessions: dict[str, ConversationalAgent] = {}

    def get_or_create_session(self, session_id: str = None, status_callback=None) -> tuple[str, ConversationalAgent]:
        if session_id and session_id in self.sessions:
            if status_callback:
                self.sessions[session_id].status_callback = status_callback
            return session_id, self.sessions[session_id]
        
        import uuid
        sid = session_id or str(uuid.uuid4())
        # 确保传入 self.settings
        self.sessions[sid] = ConversationalAgent(self.settings, status_callback)
        return sid, self.sessions[sid]
    def init_service(self):
        self.settings = Settings()
        self.user_preference = UserPreferenceMining(self.settings)
        self.rag_engine = MilvusRAGPipeline(self.settings)
        self.history_manager = ConversationHistoryManager(self.settings.HISTORY_DIR)        
        self.sessions: dict[str, ConversationalAgent] = {}
        
    

# --- 关键启动步骤 ---
engine = DeepAgentForce()
app.state.engine = engine  # 存入全局状态

app.include_router(router,prefix="/api")  # 挂载路由
setup_websocket_routes(app) # 挂载 WebSocket


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)