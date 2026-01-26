"""
工作流回调处理模块
管理状态回调和事件分发
"""

import asyncio
import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class StatusCallback:
    """状态回调管理器"""
    
    def __init__(self):
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """
        添加回调函数
        
        Args:
            callback: 回调函数，接受 (event_type: str, data: Dict[str, Any]) 参数
        """
        self.callbacks.append(callback)
        logger.debug(f"添加回调函数，当前共 {len(self.callbacks)} 个")
    
    def remove_callback(self, callback: Callable):
        """
        移除回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"移除回调函数，当前剩余 {len(self.callbacks)} 个")
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """
        触发所有回调
        
        Args:
            event_type: 事件类型 (step/token/progress/error/llm_start/llm_end)
            data: 事件数据
        """
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"回调执行错误: {e}", exc_info=True)
    
    def clear(self):
        """清空所有回调"""
        self.callbacks.clear()
        logger.debug("已清空所有回调函数")


class EventType:
    """事件类型常量"""
    
    # 工作流步骤事件
    STEP = "step"
    
    # LLM 生成事件
    TOKEN = "token"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    
    # 进度事件
    PROGRESS = "progress"
    
    # 错误和警告
    ERROR = "error"
    WARNING = "warning"
    
    # 完成事件
    DONE = "done"


class StepEvent:
    """步骤事件数据结构"""
    
    @staticmethod
    def create(step: str, title: str, description: str) -> Dict[str, Any]:
        """
        创建步骤事件
        
        Args:
            step: 步骤标识
            title: 步骤标题
            description: 步骤描述
        """
        return {
            "step": step,
            "title": title,
            "description": description
        }


class TokenEvent:
    """Token 事件数据结构"""
    
    @staticmethod
    def create(content: str, full_message: str = "") -> Dict[str, Any]:
        """
        创建 Token 事件
        
        Args:
            content: 新生成的 token
            full_message: 完整消息
        """
        return {
            "content": content,
            "full_message": full_message
        }


class ProgressEvent:
    """进度事件数据结构"""
    
    @staticmethod
    def create(current: int, total: int, description: str = "") -> Dict[str, Any]:
        """
        创建进度事件
        
        Args:
            current: 当前进度
            total: 总数
            description: 描述
        """
        return {
            "current": current,
            "total": total,
            "description": description
        }


class ErrorEvent:
    """错误事件数据结构"""
    
    @staticmethod
    def create(message: str, step: str = "", details: str = "") -> Dict[str, Any]:
        """
        创建错误事件
        
        Args:
            message: 错误信息
            step: 出错的步骤
            details: 详细信息
        """
        return {
            "message": message,
            "step": step,
            "details": details
        }


# 创建默认的日志回调
async def default_logging_callback(event_type: str, data: Dict[str, Any]):
    """默认的日志回调"""
    if event_type == EventType.STEP:
        logger.info(f"[{data.get('step')}] {data.get('title')}: {data.get('description')}")
    elif event_type == EventType.ERROR:
        logger.error(f"错误: {data.get('message')}")
    elif event_type == EventType.WARNING:
        logger.warning(f"警告: {data.get('message')}")
    elif event_type == EventType.PROGRESS:
        logger.info(f"进度: {data.get('current')}/{data.get('total')}")