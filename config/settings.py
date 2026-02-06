"""
配置管理模块
统一管理所有服务的配置信息
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache
import json
from typing import ClassVar


import json
import hashlib
from pathlib import Path
from typing import ClassVar, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""

    # --- 基础路径配置 ---
    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parent.parent

    # --- 数据目录 (Directories) ---
    # 统一定义在 data 下，方便管理
    DATA_DIR: ClassVar[Path] = PROJECT_ROOT / "data"
    
    # 历史记录 (建议改名为 DIR 如果它是一个文件夹)
    HISTORY_DIR: ClassVar[Path] = DATA_DIR / "history"
    # 上传目录
    UPLOAD_DIR: ClassVar[Path] = DATA_DIR / "uploads"
    # 向量库存储目录
    MILVUS_DIR: ClassVar[Path] = DATA_DIR / "rag_storage"
    # 技能插件目录
    SKILL_DIR: ClassVar[Path] = PROJECT_ROOT / "src" / "services" / "skills"
    # --- 数据文件 (Files) ---
    # 配置文件
    CONFIG_FILE: ClassVar[Path] = DATA_DIR / "saved_config.json"
    # 用户画像文件
    PERSON_LIKE_FILE: ClassVar[Path] = DATA_DIR / "person_like.json"
    # Milvus 数据库文件 (这是一个文件路径，用于连接字符串)
    MILVUS_DB_PATH: ClassVar[Path] = DATA_DIR / "milvus.db"

    # --- 运行时配置字段 (示例占位，防止 hash 计算报错) ---
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    LLM_URL: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None
    EMBEDDING_URL: Optional[str] = None

    def __init__(self, **kwargs):
        """初始化配置，从JSON文件加载"""
        # 先从JSON文件加载配置
        loaded_config = self._load_config_from_file()
        
        # 合并JSON配置和传入的kwargs
        merged_config = {**loaded_config, **kwargs}
        
        # 调用父类初始化
        super().__init__(**merged_config)

    @property
    def MILVUS_URL(self) -> str:
        """动态生成 Milvus 连接 URL"""
        return str(self.MILVUS_DB_PATH)

    @property
    def config_hash(self) -> str:
        """
        生成关键配置的指纹 (Hash)
        用于检测配置是否发生变化
        """
        # 使用 getattr 避免如果某些字段未定义导致报错，默认为空字符串
        key_content = (
            f"{getattr(self, 'LLM_API_KEY', '')}|"
            f"{getattr(self, 'LLM_MODEL', '')}|"
            f"{getattr(self, 'LLM_URL', '')}|"
            f"{getattr(self, 'TAVILY_API_KEY', '')}|"
            f"{getattr(self, 'FIRECRAWL_API_KEY', '')}|"
            f"{getattr(self, 'EMBEDDING_API_KEY', '')}|"
            f"{getattr(self, 'EMBEDDING_MODEL', '')}|"
        )
        return hashlib.md5(key_content.encode('utf-8')).hexdigest()

    def _ensure_directory(self, path: Path):
        """辅助方法：创建目录"""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def _ensure_json_file(self, path: Path, default_content: dict = None):
        """辅助方法：创建 JSON 文件"""
        # 先确保父目录存在
        self._ensure_directory(path.parent)
        # 如果文件不存在，写入默认内容
        if not path.exists():
            content = default_content if default_content is not None else {}
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def initialize(cls):
        """
        初始化配置并创建所有必要的文件和目录
        推荐在应用启动时 (如 main.py 开头) 调用
        """
        instance = cls()
        instance.ensure_filesystem()
        return instance

    def ensure_filesystem(self):
        """
        核心逻辑：确保所有定义的路径在文件系统中真实存在
        """
        # 1. 创建所有目录
        # 这里列出所有必须存在的目录
        directories_to_create = [
            self.DATA_DIR,
            self.HISTORY_DIR,
            self.UPLOAD_DIR,
            self.MILVUS_DIR,
        ]

        for directory in directories_to_create:
            self._ensure_directory(directory)

        # 2. 创建所有文件
        # 这里列出所有必须存在的文件
        self._ensure_json_file(self.CONFIG_FILE, default_content={})
        self._ensure_json_file(self.PERSON_LIKE_FILE, default_content={})


    # ==================== 应用基础配置 ====================
    APP_NAME: str = "IntelligentSearchAssistant"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # ==================== 服务器配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ==================== LLM 配置 ====================
    LLM_MODEL: str = ""
    LLM_URL: str = ""
    LLM_API_KEY: str = ""

    # ==================== Search 配置 ====================
    TAVILY_API_KEY: str = ""

    # ==================== FIRECRAWL 配置 ====================
    FIRECRAWL_API_KEY: str = ""

    # ==================== 执行计划配置 ====================
    MAX_PLAN_STEPS: int = Field(default=10, ge=1, le=20, description="最大计划步骤数")
    ENABLE_COMPOSITE_TASKS: bool = True

    # ==================== RAG 配置 ====================

    EMBEDDING_API_KEY: str = ""
    EMBEDDING_URL: str = ""
    EMBEDDING_MODEL: str = ""
    EMBEDDING_DIM: int = 1024
    SIMPLE_RAG: bool = True #默认为简单RAG，若使用复杂的RAG 请设置为False
    T_SCORE: float = Field(default=0.3, description="RAG检索阈值")
    RAG_URL: str = "http://localhost:8000/api/rag/query"
    MILVUS_COLLECTION: str  = "rag_chunks"

    # ==================== 内容处理配置 ====================
    MAX_CONTEXT_LENGTH: int = Field(default=8000, description="最大上下文长度")
    CONVERSATION_HISTORY_LIMIT: int = Field(default=10, description="保留的对话历史数量")
    MAX_URLS_TO_CRAWL: int = Field(default=3, description="最大爬取 URL 数量")

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==================== CORS 配置 ====================
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="允许的跨域源"
    )
    
    # ==================== Session 配置 ====================
    SESSION_TIMEOUT: int = Field(default=3600, description="Session 超时时间(秒)")
    MAX_SESSIONS: int = Field(default=1000, description="最大 Session 数量")
    def __init__(self, **kwargs):
        """初始化配置，从JSON文件加载"""
        # 先从JSON文件加载配置
        loaded_config = self._load_config_from_file()
        
        # 合并JSON配置和传入的kwargs
        merged_config = {**loaded_config, **kwargs}
        
        # 调用父类初始化
        super().__init__(**merged_config)

    def _load_config_from_file(self) -> dict:
        """从JSON文件加载配置"""
        config_data = {}
        
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    config_data=json_data
                
                print(f"✅ 成功从 {self.CONFIG_FILE} 加载配置")
                
            except json.JSONDecodeError as e:
                print(f"配置为空，不要忘记在前端进行配置。")
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}")
        else:
            print(f"⚠️ 配置文件不存在: {self.CONFIG_FILE}")
        
        return config_data

            

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # 允许额外的配置项


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用 lru_cache 确保配置只加载一次
    """
    return Settings()


# 导出配置实例
settings = get_settings()
settings.initialize()


# ==================== 配置工具函数 ====================
def get_llm_config() -> dict:
    """获取 LLM 配置"""
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_URL,
        "streaming": True,
    }


def get_planner_config() -> dict:
    """获取规划器配置"""
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_URL,
        "streaming": False,  # 规划不需要流式输出
    }


def get_search_config() -> dict:
    """获取搜索配置"""
    return {
        "tavily": {
            "api_key": settings.TAVILY_API_KEY,
        },
        "firecrawl": {
            "api_key": settings.FIRECRAWL_API_KEY,
        }
    }

