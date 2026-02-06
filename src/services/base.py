import logging
from abc import ABC, abstractmethod
from typing import Any
from config import settings  # 导入全局 settings 单例

logger = logging.getLogger(__name__)

class BaseConfigurableService(ABC):
    pass