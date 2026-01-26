"""
LLM 服务模块
封装大语言模型的调用
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import AsyncCallbackHandler
from config.settings import settings, get_llm_config, get_planner_config

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """流式输出回调处理器"""
    
    def __init__(self, status_callback: Optional[Callable] = None):
        self.tokens = []
        self.current_message = ""
        self.status_callback = status_callback
    
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> None:
        """当 LLM 开始时调用"""
        pass

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs
    ) -> None:
        """当聊天模型开始时调用"""
        if self.status_callback:
            await self.status_callback("llm_start", {
                "message": "开始生成回答..."
            })
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当 LLM 生成新 token 时调用"""
        self.tokens.append(token)
        self.current_message += token
        
        # 触发回调
        if self.status_callback:
            await self.status_callback("token", {
                "content": token,
                "full_message": self.current_message
            })
    
    async def on_llm_end(self, response, **kwargs) -> None:
        """当 LLM 结束时调用"""
        if self.status_callback:
            await self.status_callback("llm_end", {
                "message": self.current_message
            })
    
    def reset(self):
        """重置状态"""
        self.tokens = []
        self.current_message = ""


class LLMService:
    """LLM 服务类"""
    
    def __init__(self):
        """初始化 LLM 服务"""
        logger.info("✅ LLM 服务初始化成功")
    
    def create_chat_model(
        self,
        streaming: bool = True,
        callback: Optional[Callable] = None,
        use_planner_config: bool = False
    ) -> tuple[ChatOpenAI, Optional[StreamingCallbackHandler]]:
        """
        创建聊天模型
        
        Args:
            streaming: 是否启用流式输出
            callback: 状态回调函数
            use_planner_config: 是否使用规划器配置
            
        Returns:
            (模型实例, 流式处理器)
        """
        # 选择配置
        config = get_planner_config() if use_planner_config else get_llm_config()
        
        # 创建流式处理器
        streaming_handler = None
        callbacks = []
        
        if streaming and callback:
            streaming_handler = StreamingCallbackHandler(callback)
            callbacks = [streaming_handler]
        
        # 创建模型
        model = ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            streaming=streaming,
            callbacks=callbacks if callbacks else None
        )
        
        return model, streaming_handler
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        streaming: bool = True,
        callback: Optional[Callable] = None,
        use_planner_config: bool = False
    ) -> tuple[str, Optional[StreamingCallbackHandler]]:
        """
        生成回复
        
        Args:
            prompt: 用户提示词
            system_message: 系统消息
            streaming: 是否启用流式输出
            callback: 状态回调函数
            use_planner_config: 是否使用规划器配置
            
        Returns:
            (生成的内容, 流式处理器)
        """
        try:
            # 创建模型
            model, streaming_handler = self.create_chat_model(
                streaming=streaming,
                callback=callback,
                use_planner_config=use_planner_config
            )
            
            # 构建消息
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # 生成回复
            response = await model.ainvoke(messages)
            
            # 获取内容
            if streaming and streaming_handler:
                content = streaming_handler.current_message
            else:
                content = response.content
            
            return content, streaming_handler
            
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            raise
    
    async def generate_with_messages(
        self,
        messages: List[Any],
        streaming: bool = True,
        callback: Optional[Callable] = None,
        use_planner_config: bool = False
    ) -> tuple[str, Optional[StreamingCallbackHandler]]:
        """
        使用消息列表生成回复
        
        Args:
            messages: 消息列表
            streaming: 是否启用流式输出
            callback: 状态回调函数
            use_planner_config: 是否使用规划器配置
            
        Returns:
            (生成的内容, 流式处理器)
        """
        try:
            # 创建模型
            model, streaming_handler = self.create_chat_model(
                streaming=streaming,
                callback=callback,
                use_planner_config=use_planner_config
            )
            
            # 生成回复
            response = await model.ainvoke(messages)
            
            # 获取内容
            if streaming and streaming_handler:
                content = streaming_handler.current_message
            else:
                content = response.content
            
            return content, streaming_handler
            
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            raise


# 创建全局 LLM 服务实例
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取 LLM 服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service