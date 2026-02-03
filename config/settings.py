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


class Settings(BaseSettings):
    """应用配置"""

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).parent.parent
    # model config
    CONFIG_FILE: ClassVar[Path] = PROJECT_ROOT / "data" / "saved_config.json"
    # history files
    HISTORY_FILE: ClassVar[Path] = PROJECT_ROOT / "data" / "history"
    # user profile file
    PERSON_LIKE_FILE: ClassVar[Path] = PROJECT_ROOT / "data" / "person_like.json"
    # services directory
    SERVICE_DIR: ClassVar[Path] = PROJECT_ROOT / "src" / "services"

    @property
    def config_hash(self) -> str:
        """
        生成关键配置的指纹 (Hash)
        用于检测配置是否发生变化
        """
        # 1. 拼接所有影响 Agent 运行的关键字段
        # 注意：这里只包含那些"改了就需要重启模型"的字段
        # 像 LOG_LEVEL 这种改了不需要重启模型的，就不要加进来
        key_content = (
            f"{self.LLM_API_KEY}|"
            f"{self.LLM_MODEL}|"
            f"{self.LLM_URL}|"
            f"{self.TAVILY_API_KEY}|"
            f"{self.FIRECRAWL_API_KEY}|"
            f"{self.EMBEDDING_API_KEY}|"
            f"{self.EMBEDDING_MODEL}|"
        )

        return str(hash(key_content))

    @classmethod
    def ensure_data_dir(cls):
        """确保 data 目录存在"""
        data_dir = cls.PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @classmethod
    def ensure_files(cls):
        """确保所有必要的文件存在"""
        # 先确保目录存在
        cls.ensure_data_dir()
        
        # 确保 CONFIG_FILE 存在，如果不存在则创建空的 JSON 对象
        if not cls.CONFIG_FILE.exists():
            cls.CONFIG_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2))
        
        # 确保 PERSON_LIKE_FILE 存在
        if not cls.PERSON_LIKE_FILE.exists():
            cls.PERSON_LIKE_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2))
    
    @classmethod
    def initialize(cls):
        """初始化配置（推荐在应用启动时调用）"""
        cls.ensure_files()
        return cls()



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
    RAG_URL: str = "http://localhost:8000/api/graphrag/query"
    GRAPHRAG_STORAGE_DIR: str = "data/rag_graph_storage"

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
                
                # 映射JSON结构到Settings字段
                if "llm_config" in json_data:
                    llm_config = json_data["llm_config"]
                    config_data["LLM_API_KEY"] = llm_config.get("LLM_API_KEY", "")
                    config_data["LLM_URL"] = llm_config.get("LLM_URL", "")
                    config_data["LLM_MODEL"] = llm_config.get("LLM_MODEL", "")
                
                if "search_config" in json_data:
                    search_config = json_data["search_config"]
                    config_data["TAVILY_API_KEY"] = search_config.get("TAVILY_API_KEY", "")
                
                if "claw_config" in json_data:
                    claw_config = json_data["claw_config"]
                    config_data["FIRECRAWL_API_KEY"] = claw_config.get("FIRECRAWL_API_KEY", "")
                if "embedding_config" in json_data:
                    embedding_config = json_data["embedding_config"]
                    config_data["EMBEDDING_API_KEY"] = embedding_config.get("EMBEDDING_API_KEY", "")
                    config_data["EMBEDDING_URL"] = embedding_config.get("EMBEDDING_URL", "")
                    config_data["EMBEDDING_MODEL"] = embedding_config.get("EMBEDDING_MODEL", "")
                
                print(f"✅ 成功从 {self.CONFIG_FILE} 加载配置")
                
            except json.JSONDecodeError as e:
                print(f"配置为空，不要忘记在前端进行配置。")
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}")
        else:
            print(f"⚠️ 配置文件不存在: {self.CONFIG_FILE}")
        
        return config_data

    def update(self, **kwargs):
        """动态更新配置

        格式:{
            "llm_config":{
                "LLM_API_KEY":"",
                "LLM_URL":"",
                "LLM_MODEL":""
            },
            "search_config":{
                "TAVILY_API_KEY": ""
            },
            "claw_config":{
                "FIRECRAWL_API_KEY": ""
            }
        }
        """
        # 配置映射关系
        config_mapping = {
            "llm_config": {
                "LLM_API_KEY": "LLM_API_KEY",
                "LLM_URL": "LLM_URL",
                "LLM_MODEL": "LLM_MODEL"
            },
            "search_config": {
                "TAVILY_API_KEY": "TAVILY_API_KEY"
            },
            "claw_config": {
                "FIRECRAWL_API_KEY": "FIRECRAWL_API_KEY"
            }
        }
        
        for config_group, config_values in kwargs.items():
            if config_group in config_mapping:
                mapping = config_mapping[config_group]
                for key, value in config_values.items():
                    if key in mapping:
                        actual_field = mapping[key]
                        if hasattr(self, actual_field):
                            setattr(self, actual_field, value)
        
            

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

