"""
Agent 主逻辑模块
管理工作流和对话历史
"""

import logging
import operator
from typing import Optional, List, Dict, Any, Literal
from typing_extensions import TypedDict, Annotated

from langchain.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from src.services.person_like_service import UserPreferenceMining
from src.workflow.callbacks import StatusCallback
from src.workflow.nodes import (
    plan_node,
    execute_plan_node,
    synthesize_node,
    web_search_node,
    summarize_node,
    doc_search_node,
    llm_node,
    chat_node
)

logger = logging.getLogger(__name__)


# ==================== 状态定义 ====================

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List[AnyMessage], operator.add]
    search_query: str
    search_results: Dict[str, Any]
    urls_to_crawl: List[str]
    crawled_contents: List[Dict[str, Any]]
    final_summary: str
    docqa_content: str
    next_step: str
    conversation_history: List[Dict[str, str]]
    status_callback: Optional[StatusCallback]
    execution_plan: Dict[str, Any]
    collected_data: Dict[str, Any]
    user_profile: str



# ==================== 路由函数 ====================

def route_next_step(state: AgentState) -> Literal[
    "execute_plan",
    "synthesize",
    "web_search",
    "summarize",
    "doc_search",
    "llm_node",
    "chat",
    END
]:
    """
    路由函数 - 根据状态决定下一步
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_step = state.get("next_step", "end")
    
    route_map = {
        "execute_plan": "execute_plan",
        "synthesize": "synthesize",
        "web_search": "web_search",
        "summarize": "summarize",
        "doc_search": "doc_search",
        "llm_node": "llm_node",
        "chat": "chat",
    }
    
    return route_map.get(next_step, END)


# ==================== 工作流构建 ====================

def build_intelligent_workflow():
    """
    构建智能工作流
    
    Returns:
        编译后的工作流
    """
    workflow = StateGraph(AgentState)
    
    # 添加所有节点
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_plan", execute_plan_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("doc_search", doc_search_node)
    workflow.add_node("llm_node", llm_node)
    workflow.add_node("chat", chat_node)
    
    # 定义流程
    workflow.add_edge(START, "plan")
    
    # plan 节点的路由
    workflow.add_conditional_edges(
        "plan",
        route_next_step,
        {
            "execute_plan": "execute_plan",
            "web_search": "web_search",
            "doc_search": "doc_search",
            "chat": "chat",
            END: END
        }
    )
    
    # execute_plan 的路由
    workflow.add_conditional_edges(
        "execute_plan",
        route_next_step,
        {
            "synthesize": "synthesize",
            END: END
        }
    )
    
    # synthesize 完成后结束
    workflow.add_edge("synthesize", END)
    
    # 简单 web_search 流程
    workflow.add_conditional_edges(
        "web_search",
        route_next_step,
        {
            "summarize": "summarize",
            END: END
        }
    )
    
    workflow.add_edge("summarize", END)
    
    # 文档搜索流程
    workflow.add_conditional_edges(
        "doc_search",
        route_next_step,
        {
            "llm_node": "llm_node",
            END: END
        }
    )
    
    workflow.add_edge("llm_node", END)
    
    # 闲聊直接结束
    workflow.add_edge("chat", END)
    
    return workflow.compile()


# ==================== Agent 类 ====================

class ConversationalAgent:
    """
    支持上下文对话和状态回调的智能 Agent
    """
    
    def __init__(self, status_callback: Optional[StatusCallback] = None):
        """
        初始化 Agent
        
        Args:
            status_callback: 状态回调管理器
        """
        self.app = build_intelligent_workflow()
        self.conversation_history: List[Dict[str, str]] = []
        self.status_callback = status_callback
        self.user_profile = UserPreferenceMining().get_frontend_format()
        
        logger.info("✅ ConversationalAgent 初始化成功")
        
        # 可选：导出工作流图
        try:
            self._export_workflow_graph()
        except Exception as e:
            logger.warning(f"导出工作流图失败: {e}")
    
    def _export_workflow_graph(self):
        """导出工作流图为 PNG"""
        try:
            png_data = self.app.get_graph().draw_mermaid_png()
            with open("workflow_graph.png", "wb") as f:
                f.write(png_data)
            logger.info("✅ 工作流图已导出到 workflow_graph.png")
        except Exception as e:
            logger.debug(f"无法导出工作流图: {e}")
    
    async def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回回答
        
        Args:
            user_input: 用户输入的消息
            
        Returns:
            AI 的回答
        """
        try:
            # 添加用户消息到历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # 构建初始状态
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "search_query": "",
                "search_results": {},
                "urls_to_crawl": [],
                "crawled_contents": [],
                "final_summary": "",
                "docqa_content": "",
                "next_step": "",
                "conversation_history": self.conversation_history,
                "status_callback": self.status_callback,
                "execution_plan": {},
                "collected_data": {},
                "user_profile": self.user_profile.get("summary", "")
            }
            
            # 执行工作流
            logger.info(f"处理用户输入: {user_input[:50]}...")
            final_state = await self.app.ainvoke(initial_state)
            
            # 获取回答
            answer = final_state.get("final_summary", "抱歉，我无法回答这个问题。")
            
            # 添加助手回答到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            logger.info(f"生成回答: {answer[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"处理对话失败: {e}", exc_info=True)
            error_message = f"抱歉，处理您的请求时出现错误: {str(e)}"
            
            # 即使出错也记录到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": error_message
            })
            
            return error_message
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history.copy()
    
    def get_history_length(self) -> int:
        """
        获取对话历史长度
        
        Returns:
            对话轮数
        """
        return len(self.conversation_history)
    
    def trim_history(self, max_length: int = 20):
        """
        修剪对话历史，保留最近的对话
        
        Args:
            max_length: 最大保留的对话数量
        """
        if len(self.conversation_history) > max_length:
            self.conversation_history = self.conversation_history[-max_length:]
            logger.info(f"对话历史已修剪到 {max_length} 条")