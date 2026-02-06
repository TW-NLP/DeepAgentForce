import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_community.tools import ShellTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.services.person_like_service import UserPreferenceMining
from src.workflow.callbacks import StatusCallback

logger = logging.getLogger(__name__)

class ConversationalAgent():
    def __init__(self, settings, status_callback: Optional[StatusCallback] = None):
        # 显式保存 settings 和 callback
        self.settings = settings
        self.status_callback = status_callback
        self.workspace = Path(settings.SKILL_DIR)
        self.user_profile_data = UserPreferenceMining(settings).get_frontend_format()
        self.user_summary = self.user_profile_data.get("summary", "No specific preference.")
        self.exec_tool = ShellTool()
        self._instance = None
        self.exec_tool.description = (
            "允许这个shell的时候，请先看对应的SKILL.md，然后去对应的scripts里面执行对应的py文件，这个流程不要变。 "
        )
    def get_instance(self):
        """获取或创建 Deep Agent 实例（单例模式）"""
        if self._instance is None:
            self._instance = self.build_instance()
        return self._instance

    def build_instance(self):
        """
        构建 Deep Agent 实例
        """
        # 1. 初始化模型
        logger.info(f"正在使用模型: {self.settings.LLM_MODEL} 构建 Agent")
        model = init_chat_model(
            model=self.settings.LLM_MODEL,
            model_provider="openai",
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_URL
        )
        self.exec_tool = ShellTool()
        self.exec_tool.name = "shell"
        self.exec_tool.description = (
            f"Run python scripts. ALL commands must be relative to: {self.workspace}. "
            "DO NOT use absolute paths. DO NOT use 'cd' or 'ls'."
        )
        system_prompt = f"""你是一个精确执行的智能体，需要判断是否进行工具的调用，如果是闲聊，则直接回答用户的问题，如果是需要提供的技能，需要根据用户的问题来寻找一个合适的技能，并执行技能。 
**技能目录**：你可以使用的技能目录是 {self.workspace}，不允许访问其他目录，自行进行工具的调用；不允许自行进行撰写文件进行执行。
**用户**: 
# 👤 用户上下文
{self.user_summary}
"""
        
        return create_deep_agent(
            model=model,
            backend=FilesystemBackend(root_dir=str(self.settings.PROJECT_ROOT)),
            skills=[str(self.workspace)], 
            tools=[self.exec_tool],
            checkpointer=MemorySaver(),
            system_prompt=system_prompt
        )
    async def chat(self, user_input: str, thread_id: str = "default_thread") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        agent_instance = self.get_instance()
        final_response = ""
        
        try:
            # 【修改】移入 try 块，并增加日志
            if self.status_callback:
                logger.info(f"[{thread_id}] 触发 on_agent_start...")
                await self.status_callback.on_agent_start({"input": user_input})
            
            logger.info(f"[{thread_id}] 开始 Agent 流式处理: {user_input[:30]}")
            
            async for event in agent_instance.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event and len(event["messages"]) > 0:
                    last_msg = event["messages"][-1]
                    
                    # 调试日志：看看 LLM 到底返回了什么
                    # logger.info(f"Stream Event Msg Type: {type(last_msg)}")

                    # === 只有这里触发，前端才有思考框 ===
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        logger.info(f"检测到工具调用: {len(last_msg.tool_calls)} 个")
                        for tool_call in last_msg.tool_calls:
                            if self.status_callback:
                                await self.status_callback.on_tool_start(
                                    {"name": tool_call['name'], "args": tool_call['args']}
                                )
                    
                    elif isinstance(last_msg, ToolMessage):
                        logger.info("检测到工具执行结果")
                        if self.status_callback:
                            await self.status_callback.on_tool_end(
                                {"output": str(last_msg.content)[:100] + "..."}
                            )
                            
                    elif isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        final_response = last_msg.content

            if self.status_callback:
                await self.status_callback.on_agent_finish({"output": final_response})
                
            return final_response

        except Exception as e:
            logger.error(f"Chat 处理失败: {e}", exc_info=True)
            if self.status_callback:
                await self.status_callback.on_error({"message": str(e)})
            return f"系统错误: {str(e)}"
