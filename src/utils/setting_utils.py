from typing import Any, Dict
from config import settings
import json


def save_config_to_file(new_flat_config: Dict[str, str]) -> Dict[str, Any]:
    """保存配置到文件"""
    
    
    try:
        with settings.CONFIG_FILE.open('w', encoding='utf-8') as f:
            json.dump(new_flat_config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise
    
    return new_flat_config



def load_config_from_file() -> Dict[str, Any]:
    """从 JSON 文件加载配置"""
    if settings.CONFIG_FILE.exists():
        try:
            with settings.CONFIG_FILE.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            pass
    
    return {
        "llm_config": {},
        "search_config": {},
        "firecrawl_config": {},
        "embedding_config": {}
    }
