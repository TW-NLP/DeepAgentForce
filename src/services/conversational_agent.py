import logging
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
from config.settings import get_settings
from src.utils.setting_utils import load_config_from_file

logger = logging.getLogger(__name__)

def get_tenant_settings(tenant_uuid: Optional[str] = None) -> Any:
    """获取租户专属的设置对象（从 saved_config_{tenant_uuid}.json 读取）"""
    settings = get_settings()
    
    if tenant_uuid:
        # 从租户配置文件加载
        tenant_config = load_config_from_file(tenant_uuid)
        flat_config = tenant_config if isinstance(tenant_config, dict) else {}
        
        # 扁平配置键映射到 settings 属性
        if "LLM_API_KEY" in flat_config:
            settings.LLM_API_KEY = flat_config.get("LLM_API_KEY", "")
        if "LLM_URL" in flat_config:
            settings.LLM_URL = flat_config.get("LLM_URL", "")
        if "LLM_MODEL" in flat_config:
            settings.LLM_MODEL = flat_config.get("LLM_MODEL", "")
        if "EMBEDDING_API_KEY" in flat_config:
            settings.EMBEDDING_API_KEY = flat_config.get("EMBEDDING_API_KEY", "")
        if "EMBEDDING_URL" in flat_config:
            settings.EMBEDDING_URL = flat_config.get("EMBEDDING_URL", "")
        if "EMBEDDING_MODEL" in flat_config:
            settings.EMBEDDING_MODEL = flat_config.get("EMBEDDING_MODEL", "")
        if "TAVILY_API_KEY" in flat_config:
            settings.TAVILY_API_KEY = flat_config.get("TAVILY_API_KEY", "")
        if "FIRECRAWL_API_KEY" in flat_config:
            settings.FIRECRAWL_API_KEY = flat_config.get("FIRECRAWL_API_KEY", "")
        if "FIRECRAWL_URL" in flat_config:
            settings.FIRECRAWL_URL = flat_config.get("FIRECRAWL_URL", "")
    
    return settings

class ConversationalAgent():
    def __init__(self, settings, status_callback: Optional[StatusCallback] = None, tenant_uuid: Optional[str] = None):
        # 显式保存 settings 和 callback
        self.settings = settings
        self.status_callback = status_callback
        self.tenant_uuid = tenant_uuid  # 🆕 保存租户 UUID
        self.workspace = Path(settings.SKILL_DIR)
        self.user_profile_data = UserPreferenceMining(settings).get_frontend_format(tenant_uuid=tenant_uuid)  # 🆕
        self.user_summary = self.user_profile_data.get("summary", "No specific preference.")
        self.exec_tool = ShellTool()
        self._instance = None
        self.exec_tool.description = (
            f"允许这个shell的时候，请先看对应的SKILL.md，然后去对应的scripts里面执行对应的py文件，这个流程不要变,如果有新的文件生成，请统一放在{self.settings.OUTPUT_DIR}目录下。"
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
        # 🆕 如果有 tenant_uuid，使用租户配置
        if self.tenant_uuid:
            self.settings = get_tenant_settings(self.tenant_uuid)
        elif not self.settings.LLM_MODEL:
            self.settings = get_settings()

        # 1. 初始化模型
        logger.info(f"正在使用模型: {self.settings.LLM_MODEL} 构建 Agent (tenant={self.tenant_uuid})")
        model = init_chat_model(
            model=self.settings.LLM_MODEL,
            model_provider="openai",
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_URL
        )
        self.exec_tool = ShellTool()
        self.exec_tool.name = "shell"
        # 🆕 在 tool description 中包含 tenant_uuid
        tenant_info = f"tenant_uuid={self.tenant_uuid}" if self.tenant_uuid else ""
        self.exec_tool.description = (
            f"运行 Python 脚本。ALL 命令必须相对于: {self.workspace}。 "
            "DO NOT use absolute paths. DO NOT use 'cd' or 'ls'."
            "\n\n【关键】当需要执行 SKILL 技能时，必须严格遵循 SKILL.md 文件中 Execution 部分指定的命令格式。"
            "\n【关键】查看 SKILL.md 后，执行对应 scripts/ 目录下的 .py 文件。"
            + (f"\n【关键】当前租户: {self.tenant_uuid}，执行 RAG 查询时必须携带此参数。" if tenant_info else "")
        )
        system_prompt = f"""你是一个精确执行的智能体，需要判断是否进行工具的调用，如果是闲聊，则直接回答用户的问题，如果是需要提供的技能，需要根据用户的问题来寻找一个合适的技能，并执行技能。
**【关键规则】技能执行必须严格遵循 SKILL.md 中的命令格式！**
1. 首先读取对应技能的 SKILL.md 文件
2. 严格按照 SKILL.md 中 Execution 部分的命令格式执行
3. 不得自行添加、删除或修改命令参数
4. 特别注意：区分位置参数（positional）和选项参数（--flag）
**技能目录**：你可以使用的技能目录是 {self.workspace}，不允许访问其他目录。
**用户**:
# 👤 用户上下文
{self.user_summary}
""" + (f"\n**当前租户 UUID**: {self.tenant_uuid} (RAG 查询时必须携带此参数)" if self.tenant_uuid else "")
        
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
        token_buffer = ""
        
        try:
            # 【修改】移入 try 块，并增加日志
            if self.status_callback:
                logger.info(f"[{thread_id}] 触发 on_agent_start...")
                await self.status_callback.on_agent_start({"input": user_input})
            
            logger.info(f"[{thread_id}] 开始 Agent 流式处理: {user_input[:30]}")
            
            # 使用流式模式获取 token
            async for event in agent_instance.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                # event 是一个元组 (chunk, metadata)
                chunk = event[0]
                metadata = event[1]
                
                # 获取 token 内容
                if hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                    if isinstance(token, str):
                        token_buffer += token
                        # 逐个 token 发送
                        if self.status_callback:
                            await self.status_callback.on_token(token)
                    elif isinstance(token, list):
                        # 有时候 content 是列表形式
                        for item in token:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text = item.get('text', '')
                                token_buffer += text
                                if self.status_callback:
                                    await self.status_callback.on_token(text)
                
                # 获取消息用于检测工具调用
                if hasattr(metadata, 'langgraph_node') and metadata.get('langgraph_node') == 'agent':
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        for tool_call in chunk.tool_calls:
                            if self.status_callback:
                                await self.status_callback.on_tool_start(
                                    {"name": tool_call['name'], "args": tool_call['args']}
                                )
                elif hasattr(metadata, 'langgraph_node') and metadata.get('langgraph_node') == 'tools':
                    if hasattr(chunk, 'content') and self.status_callback:
                        await self.status_callback.on_tool_end(
                            {"output": str(chunk.content)[:100] + "..."}
                        )

            final_response = token_buffer
            
            if self.status_callback:
                await self.status_callback.on_agent_finish({"output": final_response})
                
            return final_response

        except Exception as e:
            logger.error(f"Chat 处理失败: {e}", exc_info=True)
            if self.status_callback:
                await self.status_callback.on_error({"message": str(e)})
            return f"系统错误: {str(e)}"
