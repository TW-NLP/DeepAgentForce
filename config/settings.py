"""
配置管理模块
统一管理所有服务的配置信息

核心配置项：
- 后端服务配置 (HOST, PORT)
- 前端服务配置 (FRONTEND_HOST, FRONTEND_PORT)
- API 地址配置 (API_BASE, WS_BASE)
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
from functools import lru_cache
import json
import hashlib


class ServerConfig(BaseSettings):
    """服务器配置 - 前后端统一配置"""

    # ==================== 后端服务配置 ====================
    HOST: str = Field(default="0.0.0.0", description="后端服务监听地址")
    PORT: int = Field(default=8000, ge=1, le=65535, description="后端服务监听端口")

    # ==================== 前端服务配置 ====================
    FRONTEND_HOST: str = Field(default="127.0.0.1", description="前端服务地址")
    FRONTEND_PORT: int = Field(default=8080, ge=1, le=65535, description="前端服务端口")

    @property
    def API_BASE(self) -> str:
        """动态生成后端 API 地址（包含 /api 前缀）"""
        return f"http://{self.HOST if self.HOST != '0.0.0.0' else '127.0.0.1'}:{self.PORT}/api"

    @property
    def WS_BASE(self) -> str:
        """动态生成 WebSocket 地址"""
        return f"ws://{self.HOST if self.HOST != '0.0.0.0' else '127.0.0.1'}:{self.PORT}/ws/stream"

    @property
    def FRONTEND_BASE(self) -> str:
        """前端服务地址"""
        return f"http://{self.FRONTEND_HOST}:{self.FRONTEND_PORT}"

    @property
    def server_info(self) -> dict:
        """返回供前端使用的服务器信息"""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "api_base": self.API_BASE,
            "ws_base": self.WS_BASE,
            "frontend_host": self.FRONTEND_HOST,
            "frontend_port": self.FRONTEND_PORT,
            "frontend_base": self.FRONTEND_BASE,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class Settings(ServerConfig):
    """应用完整配置"""

    # ==================== 基础路径配置 ====================
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # ==================== 数据目录 ====================
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    HISTORY_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "history")
    UPLOAD_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "uploads")
    MILVUS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "rag_storage")
    SKILL_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "src" / "services" / "skills")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "outputs")

    # ==================== 配置文件 ====================
    CONFIG_FILE: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "saved_config.json")
    PERSON_LIKE_FILE: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "person_like.json")
    MILVUS_DB_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "milvus.db")

    @property
    def MILVUS_URL(self) -> str:
        """Milvus 数据库连接路径"""
        return str(self.MILVUS_DB_PATH)

    # ==================== 应用信息 ====================
    APP_NAME: str = "DeepAgentForce"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ==================== LLM 配置 ====================
    LLM_MODEL: str = ""
    LLM_URL: str = ""
    LLM_API_KEY: str = ""

    # ==================== 搜索配置 ====================
    TAVILY_API_KEY: str = ""
    FIRECRAWL_API_KEY: str = ""

    # ==================== RAG 配置 ====================
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_URL: str = ""
    EMBEDDING_MODEL: str = ""
    EMBEDDING_DIM: int = 1024
    SIMPLE_RAG: bool = True
    T_SCORE: float = Field(default=0.3, description="RAG 检索阈值")
    RAG_URL: str = ""  # 动态生成
    MILVUS_COLLECTION: str = "rag_chunks"

    @property
    def RAG_API_URL(self) -> str:
        """动态生成 RAG 查询地址"""
        return f"{self.API_BASE}/rag/query"

    # ==================== 执行计划配置 ====================
    MAX_PLAN_STEPS: int = Field(default=10, ge=1, le=20, description="最大计划步骤数")
    ENABLE_COMPOSITE_TASKS: bool = True

    # ==================== 内容处理配置 ====================
    MAX_CONTEXT_LENGTH: int = Field(default=8000, description="最大上下文长度")
    CONVERSATION_HISTORY_LIMIT: int = Field(default=10, description="保留的对话历史数量")
    MAX_URLS_TO_CRAWL: int = Field(default=3, description="最大爬取 URL 数量")

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==================== CORS 配置 ====================
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="允许的跨域源"
    )

    # ==================== Session 配置 ====================
    SESSION_TIMEOUT: int = Field(default=3600, description="Session 超时时间(秒)")
    MAX_SESSIONS: int = Field(default=1000, description="最大 Session 数量")

    def __init__(self, **kwargs):
        """初始化配置，从 JSON 文件加载"""
        super().__init__(**kwargs)
        self._ensure_directories()
        self._load_from_file()

    def _ensure_directories(self):
        """确保必要目录存在"""
        directories = [
            self.DATA_DIR,
            self.HISTORY_DIR,
            self.UPLOAD_DIR,
            self.MILVUS_DIR,
            self.OUTPUT_DIR,
        ]
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

    def _load_from_file(self):
        """从配置文件加载"""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # 更新配置项
                for key, value in config_data.items():
                    if hasattr(self, key) and key not in ['HOST', 'PORT', 'FRONTEND_HOST', 'FRONTEND_PORT']:
                        setattr(self, key, value)

                print(f"✅ 成功从 {self.CONFIG_FILE} 加载配置")
            except json.JSONDecodeError:
                print("⚠️ 配置文件为空，请在前端进行配置")
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}")
        else:
            print(f"⚠️ 配置文件不存在: {self.CONFIG_FILE}")

    def _ensure_json_file(self, path: Path, default_content: dict = None):
        """确保 JSON 文件存在"""
        if not path.exists():
            content = default_content if default_content is not None else {}
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')

    @property
    def config_hash(self) -> str:
        """生成关键配置的指纹 (Hash)"""
        key_content = (
            f"{self.LLM_API_KEY}|{self.LLM_MODEL}|{self.LLM_URL}|"
            f"{self.TAVILY_API_KEY}|{self.FIRECRAWL_API_KEY}|"
            f"{self.EMBEDDING_API_KEY}|{self.EMBEDDING_MODEL}"
        )
        return hashlib.md5(key_content.encode('utf-8')).hexdigest()

    def save_to_file(self, config_data: dict):
        """保存配置到文件"""
        # 过滤掉服务器配置和路径配置
        excluded_keys = {
            'PROJECT_ROOT', 'DATA_DIR', 'HISTORY_DIR', 'UPLOAD_DIR', 'MILVUS_DIR',
            'SKILL_DIR', 'CONFIG_FILE', 'PERSON_LIKE_FILE', 'MILVUS_DB_PATH',
            'MILVUS_URL', 'API_BASE', 'WS_BASE', 'FRONTEND_BASE', 'server_info',
            'config_hash', 'RAG_API_URL', 'APP_NAME', 'APP_VERSION', 'DEBUG',
            'LOG_LEVEL', 'LOG_FORMAT', 'CORS_ORIGINS', 'SESSION_TIMEOUT', 'MAX_SESSIONS'
        }

        filtered_config = {k: v for k, v in config_data.items() if k not in excluded_keys}

        self.CONFIG_FILE.write_text(
            json.dumps(filtered_config, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"✅ 配置已保存到 {self.CONFIG_FILE}")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用 lru_cache 确保配置只加载一次
    """
    return Settings()


def get_llm_config() -> dict:
    """获取 LLM 配置"""
    settings = get_settings()
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_URL,
        "streaming": True,
    }


def get_planner_config() -> dict:
    """获取规划器配置"""
    settings = get_settings()
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_URL,
        "streaming": False,
    }


def get_search_config() -> dict:
    """获取搜索配置"""
    settings = get_settings()
    return {
        "tavily": {"api_key": settings.TAVILY_API_KEY},
        "firecrawl": {"api_key": settings.FIRECRAWL_API_KEY}
    }


# ==================== 导出配置实例 ====================
settings = get_settings()

