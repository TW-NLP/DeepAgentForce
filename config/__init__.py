"""
配置模块
"""

from config.settings import settings, get_settings, validate_settings
from config.prompts import prompts

__all__ = [
    "settings",
    "get_settings",
    "validate_settings",
    "prompts"
]