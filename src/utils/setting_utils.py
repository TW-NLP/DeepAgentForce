from typing import Any, Dict, Optional
from config import settings
import json


def save_config_to_file(
    new_flat_config: Dict[str, str],
    tenant_uuid: Optional[str] = None
) -> Dict[str, Any]:
    """
    保存配置到文件

    Args:
        new_flat_config: 新的配置数据
        tenant_uuid: 租户UUID 🆕
    """
    try:
        # 🆕 多租户：使用租户专属配置文件
        config_file = settings.get_tenant_config_file(tenant_uuid)
        with config_file.open('w', encoding='utf-8') as f:
            json.dump(new_flat_config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise

    return new_flat_config


def load_config_from_file(tenant_uuid: Optional[str] = None) -> Dict[str, Any]:
    """
    从 JSON 文件加载配置

    Args:
        tenant_uuid: 租户UUID 🆕
    """
    # 🆕 多租户：使用租户专属配置文件
    config_file = settings.get_tenant_config_file(tenant_uuid)
    if config_file.exists():
        try:
            with config_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "llm_config": {},
        "search_config": {},
        "firecrawl_config": {},
        "embedding_config": {}
    }
