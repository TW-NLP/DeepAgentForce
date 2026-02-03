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
from src.services.base import BaseConfigurableService
from src.services.person_like_service import UserPreferenceMining
from src.workflow.callbacks import StatusCallback

logger = logging.getLogger(__name__)

class ConversationalAgent(BaseConfigurableService):
    """
    基于 DeepAgents 重构的智能 Agent
    """    
    def __init__(self, status_callback: Optional[StatusCallback] = None):
        super().__init__()
        
        self.status_callback = status_callback
        self.workspace = self.settings.SERVICE_DIR
        # 1. 准备用户画像
        self.user_profile_data = UserPreferenceMining().get_frontend_format()
        self.user_summary = self.user_profile_data.get("summary", "No specific preference.")
        # 2. 基础设施工具 (Shell)
        self.exec_tool = ShellTool()
        self.exec_tool.description = (
            "Execute shell commands. Use this ONLY when a Skill documentation "
            "instructs you to run a specific python script."
        )
        # 3. 确定技能目录
        self.skills_dir = self.workspace / "skills"

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
        system_prompt = f"""你是一个精确执行的智能体，需要判断是否进行工具的调用，如果是闲聊，则直接回答用户的问题，如果是需要使用技能，则严格按照下面的规范执行。 
# 🛠 执行规范（严格）

1. **工作目录**：你当前的工作目录是 {self.workspace}。  
2. **命令格式**：**只能**使用以下格式：  
   python skills/<skill-name>/scripts/<script_name>.py --query "your query"  
3. **禁止探索**：不要使用 ls 查看文件。不要使用 cd 切换目录。  
4. **禁止绝对路径**：绝不能使用以 /Users/... 开头的路径，只能使用以 skills/ 开头的相对路径。  

# 👤 用户上下文
{self.user_summary}
"""
        
        return create_deep_agent(
            model=model,
            backend=FilesystemBackend(root_dir=str(self.workspace)),
            skills=[str(self.skills_dir)], 
            tools=[self.exec_tool],
            checkpointer=MemorySaver(),
            system_prompt=system_prompt
        )
    
    async def chat(self, user_input: str, thread_id: str = "default_thread") -> str:
        """
        处理对话，兼容旧接口，并适配 StatusCallback
        """
        config = {"configurable": {"thread_id": thread_id}}
        agent_instance=self.get_instance()
        
        # 触发回调：开始
        if self.status_callback:
            # 模拟旧版回调结构
            await self.status_callback.on_agent_start({"input": user_input})
        final_response = ""
        try:
            logger.info(f"处理用户输入: {user_input[:50]}...")
            
            # 使用 stream 来获取中间步骤，以触发回调
            async for event in agent_instance.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event and len(event["messages"]) > 0:
                    last_msg = event["messages"][-1]
                    
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            action_name = tool_call['name']
                            args = tool_call['args']
                            logger.info(f"Agent 正在调用工具: {action_name}")
                            
                            if self.status_callback:
                                # 模拟发送状态更新
                                await self.status_callback.on_tool_start(
                                    {"name": action_name, "args": args}
                                )
                    
                    elif isinstance(last_msg, ToolMessage):
                        logger.info(f"工具执行完成: {last_msg.name}")
                        if self.status_callback:
                            await self.status_callback.on_tool_end(
                                {"output": str(last_msg.content)[:200] + "..."}
                            )               
                    elif isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        final_response = last_msg.content
            # 触发回调：结束
            if self.status_callback:
                await self.status_callback.on_agent_finish({"output": final_response})       
            logger.info(f"生成回答: {final_response}...")
            return final_response
        except Exception as e:
            logger.error(f"处理对话失败: {e}", exc_info=True)

            return f"系统错误: {str(e)}"