"""
配置管理模块
统一管理所有服务的配置信息

核心配置项：
- 后端服务配置 (HOST, PORT)
- 前端服务配置 (FRONTEND_HOST, FRONTEND_PORT)
- API 地址配置 (API_BASE, WS_BASE)
"""

import os
import logging
import sys
import tempfile
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
from functools import lru_cache
import json
import hashlib
import socket


def _get_local_ip() -> str:
    """自动获取本机局域网 IP（用于对外暴露的地址）"""
    try:
        # 连接到一个外部地址来确定本机出口 IP（不实际发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# 启动时自动检测本机 IP，避免硬编码 127.0.0.1
_LOCAL_IP = _get_local_ip()
_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", _SOURCE_ROOT))
_EXECUTABLE_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _SOURCE_ROOT


def _get_frozen_data_dir() -> Path:
    """为打包后的桌面应用选择可写的数据目录。"""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "DeepAgentForce"
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "DeepAgentForce"
        return Path.home() / "AppData" / "Roaming" / "DeepAgentForce"
    return Path.home() / ".local" / "share" / "DeepAgentForce"


_DEFAULT_DATA_DIR = _get_frozen_data_dir() if getattr(sys, "frozen", False) else (_SOURCE_ROOT / "data")
_APP_DATA_DIR = Path(os.getenv("DEEPAGENTFORCE_DATA_DIR", str(_DEFAULT_DATA_DIR))).expanduser()
_APP_LOG_DIR = _APP_DATA_DIR / "logs"
_APP_LOG_FILE = _APP_LOG_DIR / "startup.log"
_BUNDLE_ENV_FILE = _EXECUTABLE_ROOT / ".env"
_APP_ENV_FILE = os.getenv(
    "DEEPAGENTFORCE_ENV_FILE",
    str(_BUNDLE_ENV_FILE if _BUNDLE_ENV_FILE.exists() else (_APP_DATA_DIR / ".env"))
)


def _setup_bootstrap_logging() -> None:
    """尽早把启动日志写入文件，方便 GUI 包静默退出时排查问题。"""
    log_dir = _APP_LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 某些受限环境下用户目录不可写，回退到临时目录保证不会影响启动。
        log_dir = Path(tempfile.gettempdir()) / "DeepAgentForce" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    has_file_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).parent == log_dir:
            has_file_handler = True
            break

    if not has_file_handler:
        try:
            file_handler = logging.FileHandler(log_dir / "startup.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
        except Exception:
            # 文件日志不可用时，至少别影响程序启动。
            pass

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        for handler in root_logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught_exception


_setup_bootstrap_logging()


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
        host = _LOCAL_IP if self.HOST == "0.0.0.0" else self.HOST
        return f"http://{host}:{self.PORT}/api"

    @property
    def WS_BASE(self) -> str:
        """动态生成 WebSocket 地址"""
        host = _LOCAL_IP if self.HOST == "0.0.0.0" else self.HOST
        return f"ws://{host}:{self.PORT}/ws/stream"

    @property
    def FRONTEND_BASE(self) -> str:
        """前端服务地址"""
        host = _LOCAL_IP if self.FRONTEND_HOST in ("127.0.0.1", "0.0.0.0", "") else self.FRONTEND_HOST
        return f"http://{host}:{self.FRONTEND_PORT}"

    @property
    def server_info(self) -> dict:
        """返回供前端使用的服务器信息"""
        # 用 API_BASE 反推对外暴露的真实 host（去掉 /api 后缀）
        accessible_host = self.API_BASE.rsplit("/api", 1)[0].replace("http://", "").replace("https://", "")
        return {
            "host": accessible_host,
            "port": self.PORT,
            "api_base": self.API_BASE,
            "ws_base": self.WS_BASE,
            "frontend_host": accessible_host,
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
    PROJECT_ROOT: Path = _SOURCE_ROOT if not getattr(sys, "frozen", False) else _BUNDLE_ROOT

    # ==================== 数据目录 ====================
    DATA_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR)
    HISTORY_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR / "history")
    UPLOAD_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR / "uploads")
    CHROMA_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR / "rag_storage")
    SKILL_DIR: Path = Field(default_factory=lambda: _BUNDLE_ROOT / "src" / "services" / "skills")
    USER_SKILL_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR / "skill")
    OUTPUT_DIR: Path = Field(default_factory=lambda: _APP_DATA_DIR / "outputs")

    # ==================== 配置文件 ====================
    CONFIG_FILE: Path = Field(default_factory=lambda: _APP_DATA_DIR / "saved_config.json")
    PERSON_LIKE_FILE: Path = Field(default_factory=lambda: _APP_DATA_DIR / "person_like.json")
    CHROMA_COLLECTION: str = Field(default="rag_chunks")

    @property
    def DB_URL(self) -> str:
        """数据库连接 URL"""
        return f"sqlite:///{self.SQLITE_DB_PATH}"

    # Milvus removed; use CHROMA_DIR / CHROMA_COLLECTION for vector storage

    # ==================== 应用信息 ====================
    APP_NAME: str = "DeepAgentForce"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _coerce_debug_flag(cls, value):
        """允许把常见的字符串调试标志映射为布尔值，避免启动时因环境变量崩溃。"""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "prod", "production", "release", ""}:
                return False
        return value

    # ==================== LLM 配置 ====================
    LLM_MODEL: str = ""
    LLM_URL: str = ""
    LLM_API_KEY: str = ""

    @property
    def LLM_BASE_URL(self) -> str:
        """LLM API 的 base URL（自动去掉 /chat/completions 后缀，避免路径重复）"""
        if not self.LLM_URL:
            return ""
        return self.LLM_URL.rstrip("/").replace("/chat/completions", "")

    # ==================== 校对专用模型配置 ====================
    PROOFREAD_USE_DEDICATED: bool = False
    PROOFREAD_API_URL: str = ""
    PROOFREAD_API_KEY: str = ""
    PROOFREAD_MODEL: str = ""

    @property
    def PROOFREAD_BASE_URL(self) -> str:
        """校对专用模型 API 的 base URL（自动去掉 /chat/completions 后缀，保留 /v1）"""
        if not self.PROOFREAD_API_URL:
            return ""
        url = self.PROOFREAD_API_URL.rstrip("/")
        url = url.replace("/chat/completions", "")
        return url

    # ==================== 搜索配置 ====================
    TAVILY_API_KEY: str = ""
    FIRECRAWL_API_KEY: str = ""

    # ==================== RAG 配置 ====================
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_URL: str = ""
    EMBEDDING_MODEL: str = ""

    @property
    def EMBEDDING_BASE_URL(self) -> str:
        """Embedding API 的 base URL（自动去掉 /embeddings 后缀，避免路径重复）"""
        if not self.EMBEDDING_URL:
            return ""
        return self.EMBEDDING_URL.rstrip("/").replace("/embeddings", "")
    EMBEDDING_DIM: int = 1024
    SIMPLE_RAG: bool = True
    T_SCORE: float = Field(default=0.3, description="RAG 检索阈值")
    RAG_URL: str = ""  # 动态生成

    # ==================== RAG 高级配置 ====================
    # Stage1: 关键词召回（BM25）
    ENABLE_KEYWORD_SEARCH: bool = Field(default=True, description="启用关键词召回（BM25）")
    KEYWORD_TOP_K: int = Field(default=10, description="关键词召回返回数量")

    # Stage2: Rerank 重排
    ENABLE_RERANK: bool = Field(default=False, description="启用 Rerank 重排")
    RERANK_API_URL: str = Field(default="", description="Rerank 模型 API 地址")
    RERANK_API_KEY: str = Field(default="", description="Rerank 模型 API Key")
    RERANK_MODEL: str = Field(default="", description="Rerank 模型名称")
    RERANK_TOP_N: int = Field(default=5, description="Rerank 后返回数量")
    RERANK_BATCH_SIZE: int = Field(default=20, description="Rerank 每批送入的候选数量")

    # Query Rewrite 多路投票
    ENABLE_QUERY_REWRITE: bool = Field(default=False, description="启用 Query Rewrite 多路投票")
    QUERY_REWRITE_NUM: int = Field(default=3, description="Query Rewrite 生成子问题数量")

    @property
    def RAG_API_URL(self) -> str:
        """
        动态生成 RAG 查询地址
        内部服务调用走 localhost（同服务器内部 HTTP 调用无需绕公网）
        """
        return f"http://127.0.0.1:{self.PORT}/api/rag/query"

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

    # ==================== 数据库配置 ====================
    SQLITE_DB_PATH: Path = Field(default_factory=lambda: _APP_DATA_DIR / "deepagentforce.db")

    # ==================== JWT 配置 ====================
    JWT_SECRET_KEY: str = Field(default="your-secret-key-change-in-production", description="JWT 密钥")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT 算法")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="访问令牌过期时间(分钟)")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="刷新令牌过期时间(天)")

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
            self.CHROMA_DIR,
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
                immutable_keys = {
                    'PROJECT_ROOT', 'DATA_DIR', 'HISTORY_DIR', 'UPLOAD_DIR', 'CHROMA_DIR',
                    'SKILL_DIR', 'USER_SKILL_DIR', 'OUTPUT_DIR', 'CONFIG_FILE', 'PERSON_LIKE_FILE',
                    'SQLITE_DB_PATH'
                }
                for key, value in config_data.items():
                    if hasattr(self, key) and key not in ['HOST', 'PORT', 'FRONTEND_HOST', 'FRONTEND_PORT'] and key not in immutable_keys:
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
            'PROJECT_ROOT', 'DATA_DIR', 'HISTORY_DIR', 'UPLOAD_DIR', 'CHROMA_DIR',
            'SKILL_DIR', 'USER_SKILL_DIR', 'CONFIG_FILE', 'PERSON_LIKE_FILE',
            'API_BASE', 'WS_BASE', 'FRONTEND_BASE', 'server_info',
            'config_hash', 'RAG_API_URL', 'APP_NAME', 'APP_VERSION', 'DEBUG',
            'LOG_LEVEL', 'LOG_FORMAT', 'CORS_ORIGINS', 'SESSION_TIMEOUT', 'MAX_SESSIONS',
            'SQLITE_DB_PATH'
        }

        filtered_config = {k: v for k, v in config_data.items() if k not in excluded_keys}

        self.CONFIG_FILE.write_text(
            json.dumps(filtered_config, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"✅ 配置已保存到 {self.CONFIG_FILE}")

    # 🆕 多租户路径支持（使用 tenant_uuid）

    def get_tenant_history_dir(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的历史目录（使用 tenant_uuid）"""
        if tenant_uuid is None:
            tenant_uuid = "default"
        return self.HISTORY_DIR / str(tenant_uuid)

    def get_tenant_upload_dir(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的上传目录（使用 tenant_uuid）"""
        if tenant_uuid is None:
            tenant_uuid = "default"
        return self.UPLOAD_DIR / str(tenant_uuid)

    def get_tenant_output_dir(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的输出目录（使用 tenant_uuid）"""
        if tenant_uuid is None:
            tenant_uuid = "default"
        tenant_dir = self.OUTPUT_DIR / str(tenant_uuid)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir

    def get_tenant_config_file(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的配置文件（使用 tenant_uuid）"""
        if tenant_uuid is None:
            return self.CONFIG_FILE
        return self.CONFIG_FILE.parent / f"saved_config_{tenant_uuid}.json"

    def get_tenant_person_like_file(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的用户画像文件（使用 tenant_uuid）"""
        if tenant_uuid is None:
            return self.PERSON_LIKE_FILE
        return self.PERSON_LIKE_FILE.parent / f"person_like_{tenant_uuid}.json"

    def get_user_skill_dir(self, tenant_uuid: str) -> Path:
        """获取用户专属的 Skills 目录（按 tenant_uuid 隔离）"""
        user_dir = self.USER_SKILL_DIR / tenant_uuid
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_builtin_skill_dir(self) -> Path:
        """获取内置 Skills 目录（所有用户可见）"""
        return self.SKILL_DIR

    class Config:
        env_file = _APP_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"

    @field_validator("SQLITE_DB_PATH", mode="before")
    @classmethod
    def _resolve_sqlite_db_path(cls, value):
        """允许在 .env 中写相对路径，并按项目/应用根目录解析。"""
        if value is None or str(value).strip() == "":
            return _APP_DATA_DIR / "deepagentforce.db"

        path = Path(str(value)).expanduser()
        if getattr(sys, "frozen", False):
            # 桌面打包版统一使用可写的应用数据目录，避免把 SQLite 指到只读的 app bundle 内。
            return _APP_DATA_DIR / path.name

        base_dir = _APP_DATA_DIR if getattr(sys, "frozen", False) else _SOURCE_ROOT
        return (base_dir / path).resolve()


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
        "base_url": settings.LLM_BASE_URL,
        "streaming": True,
    }


def get_planner_config() -> dict:
    """获取规划器配置"""
    settings = get_settings()
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_BASE_URL,
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
