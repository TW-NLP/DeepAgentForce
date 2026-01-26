"""
配置管理模块
统一管理所有服务的配置信息
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import json
from typing import ClassVar


class Settings(BaseSettings):
    """应用配置"""
    # 加载已有的配置
    CONFIG_FILE: ClassVar[Path] = Path("config/saved_config.json")


    
    # ==================== 应用基础配置 ====================
    APP_NAME: str = "IntelligentSearchAssistant"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # ==================== 服务器配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ==================== LLM 配置 ====================
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""

    # ==================== Search 配置 ====================
    TAVILY_API_KEY: str = ""

    # ==================== FIRECRAWL 配置 ====================
    FIRECRAWL_API_KEY: str = ""

    # ==================== 执行计划配置 ====================
    MAX_PLAN_STEPS: int = Field(default=10, ge=1, le=20, description="最大计划步骤数")
    ENABLE_COMPOSITE_TASKS: bool = True

    # ==================== RAG 配置 ====================
    SIMPLE_RAG: bool = True #默认为简单RAG，若使用复杂的RAG 请设置为False
    ThRESHOLD_SCORE: float = Field(default=0.4, ge=0.0, le=1.0, description="RAG检索阈值")
    RAG_URL: str = "http://localhost:8000/api/graphrag/query"

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
                    config_data["LLM_BASE_URL"] = llm_config.get("LLM_URL", "")
                    config_data["LLM_MODEL"] = llm_config.get("LLM_MODEL", "")
                
                if "search_config" in json_data:
                    search_config = json_data["search_config"]
                    config_data["TAVILY_API_KEY"] = search_config.get("TAVILY_API_KEY", "")
                
                if "claw_config" in json_data:
                    claw_config = json_data["claw_config"]
                    config_data["FIRECRAWL_API_KEY"] = claw_config.get("FIRECRAWL_API_KEY", "")
                
                print(f"✅ 成功从 {self.CONFIG_FILE} 加载配置")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析错误: {e}")
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
                "LLM_URL": "LLM_BASE_URL",
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


# ==================== 配置验证 ====================
def validate_settings():
    """验证关键配置项"""
    errors = []
    
    if not settings.LLM_API_KEY:
        errors.append("LLM_API_KEY 未配置")
    
    if not settings.TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY 未配置")

    
    if errors:
        raise ValueError(f"配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# ==================== 配置工具函数 ====================
def get_llm_config() -> dict:
    """获取 LLM 配置"""
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_BASE_URL,
        "streaming": True,
    }


def get_planner_config() -> dict:
    """获取规划器配置"""
    return {
        "model": settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_BASE_URL,
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

