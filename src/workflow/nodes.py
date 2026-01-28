"""
工作流节点模块
定义所有工作流节点函数
"""

import json
import logging
from typing import Dict, Any, Optional
from langchain.messages import HumanMessage, AIMessage

from config.settings import settings
from config.prompts import prompts
from src.services.llm_service import get_llm_service
from src.services.search_service import get_search_service
from src.services.rag_service import get_rag_service
from src.workflow.callbacks import EventType, StepEvent

logger = logging.getLogger(__name__)


# 类型定义
AgentState = Dict[str, Any]


async def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    智能任务规划节点 - 分析任务并制定执行计划
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="planning",
            title="🤔 分析任务",
            description="正在制定执行计划..."
        ))
    
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        conversation_history = state.get("conversation_history", [])
        user_profile = state.get("user_profile", "暂无用户偏好信息")
        
        # 格式化对话历史
        history_context = prompts.format_history_context(
            conversation_history,
            limit=settings.CONVERSATION_HISTORY_LIMIT
        )
        
        # 构建规划提示词
        plan_prompt = prompts.TASK_PLANNING.format(
            history_context=history_context,
            user_profile=user_profile,
            query=query
        )
        
        # 调用 LLM 生成计划
        llm_service = get_llm_service()
        try:
            content, _ = await llm_service.generate(
                prompt=plan_prompt,
                streaming=False,
                use_planner_config=True
            )
            
            # 清理可能的 markdown 代码块
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            plan = json.loads(content.strip())
            
            if callback:
                steps_desc = "\n".join([
                    f"  {i+1}. {s['action']}: {s['reason']}"
                    for i, s in enumerate(plan.get('steps', []))
                ])
                await callback.emit(EventType.STEP, StepEvent.create(
                    step="plan_decided",
                    title="✓ 规划完成",
                    description=f"任务类型: {plan.get('task_type', 'unknown')}\n执行步骤:\n{steps_desc}"
                ))
            
            # 判断下一步
            if plan.get("task_type") == "composite":
                next_step = "execute_plan"
            elif len(plan.get("steps", [])) == 1:
                # 单步任务直接路由
                single_step = plan["steps"][0]["action"]
                next_step = single_step
            else:
                next_step = "execute_plan"
            
            logger.info(f"[规划] 任务类型: {plan.get('task_type')}, 下一步: {next_step}")
            
            return {
                "search_query": query,
                "execution_plan": plan,
                "next_step": next_step
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"规划解析失败: {e}, 响应: {content}")
            # 降级为聊天模式
            return {
                "search_query": query,
                "next_step": "chat"
            }
        except Exception as e:
            logger.error(f"规划失败: {e}")
            return {
                "search_query": query,
                "next_step": "chat"
            }
    
    return {"next_step": "end"}


async def execute_plan_node(state: AgentState) -> Dict[str, Any]:
    """
    执行多步骤计划节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    plan = state.get("execution_plan", {})
    
    if not plan or "steps" not in plan:
        return {"next_step": "chat"}
    
    steps = plan["steps"]
    logger.info(f'正在执行复合任务，步骤: {[s["action"] for s in steps]}')
    
    collected_data = {
        "web_results": [],
        "doc_results": []
    }
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="execute_start",
            title="🚀 开始执行计划",
            description=f"共 {len(steps)} 个步骤"
        ))
    
    # 执行每个步骤
    for i, step in enumerate(steps, 1):
        action = step["action"]
        query = step.get("query", state["search_query"])
        
        if callback:
            await callback.emit(EventType.STEP, StepEvent.create(
                step=f"execute_{i}",
                title=f"📋 执行步骤 {i}/{len(steps)}",
                description=f"{step['reason']}: {query}"
            ))
        
        if action == "web_search":
            # 执行网络搜索
            try:
                search_service = get_search_service()
                result = await search_service.search_and_crawl(
                    query=query,
                    num_urls=settings.MAX_URLS_TO_CRAWL
                )
                
                collected_data["web_results"].append({
                    "query": query,
                    "reason": step["reason"],
                    "content": result.get("crawled_contents", [])
                })
                
                if callback:
                    await callback.emit(EventType.STEP, StepEvent.create(
                        step=f"web_search_done_{i}",
                        title=f"✓ 网络搜索完成",
                        description=f"获取了 {len(result.get('crawled_contents', []))} 个网页内容"
                    ))
            except Exception as e:
                logger.error(f"网络搜索失败: {e}")
                if callback:
                    await callback.emit(EventType.ERROR, {
                        "step": f"web_search_error_{i}",
                        "message": f"搜索失败: {str(e)}"
                    })
        
        elif action == "doc_search":
            # 执行文档搜索
            try:
                rag_service = get_rag_service()
                result = await rag_service.search_and_format(
                    query=query
                )
                logger.info(f"[文档搜索] 查询: {query}, 响应: {result}")
                
                collected_data["doc_results"].append({
                    "query": query,
                    "reason": step["reason"],
                    "content": result.get("formatted_content", "")
                })
                
                if callback:
                    await callback.emit(EventType.STEP, StepEvent.create(
                        step=f"doc_search_done_{i}",
                        title=f"✓ 文档搜索完成",
                        description=f"检索到相关文档"
                    ))
            except Exception as e:
                logger.error(f"文档搜索失败: {e}")
                if callback:
                    await callback.emit(EventType.ERROR, {
                        "step": f"doc_search_error_{i}",
                        "message": f"文档检索失败: {str(e)}"
                    })
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="execute_complete",
            title="✓ 计划执行完成",
            description=f"已完成 {len(steps)} 个步骤的数据收集"
        ))
    
    return {
        "collected_data": collected_data,
        "messages": [AIMessage(content=f"已完成 {len(steps)} 个步骤的数据收集")],
        "next_step": "synthesize" if plan.get("final_action") == "synthesize" else "end"
    }


async def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """
    综合分析节点 - 整合多源数据生成建议
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    logger.info(f"进入综合分析节点, 问题: {state['search_query']}")
    callback = state.get("status_callback")
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="synthesizing",
            title="🧠 综合分析",
            description="正在整合信息并生成建议..."
        ))
    
    query = state["search_query"]
    collected_data = state.get("collected_data", {})
    conversation_history = state.get("conversation_history", [])
    
    # 格式化上下文
    history_context = prompts.format_history_context(conversation_history)

    logger.info(f"综合分析节点, collected_data: {collected_data}")

    web_context = prompts.format_web_context(collected_data.get("web_results", []))
    doc_context = prompts.format_doc_context(collected_data.get("doc_results", []))
    
    # 构建综合提示词
    prompt = prompts.SYNTHESIS.format(
        history_context=history_context,
        query=query,
        web_context=web_context,
        doc_context=doc_context
    )
    
    # 流式生成
    llm_service = get_llm_service()
    
    async def status_callback_wrapper(event_type: str, data: Dict[str, Any]):
        if callback:
            await callback.emit(event_type, data)
    
    content, _ = await llm_service.generate(
        prompt=prompt,
        streaming=True,
        callback=status_callback_wrapper
    )
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="synthesize_complete",
            title="✓ 分析完成",
            description="已生成综合建议"
        ))
    
    return {
        "final_summary": content,
        "messages": [AIMessage(content=content)],
        "next_step": "end"
    }


async def web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    网络搜索节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    query = state["search_query"]
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="web_search",
            title="🔍 网络搜索",
            description=f"正在搜索: {query}"
        ))
    
    # 调用搜索服务
    search_service = get_search_service()
    result = await search_service.search_and_crawl(query)
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="search_complete",
            title="✓ 搜索完成",
            description=f"找到 {len(result.get('crawled_contents', []))} 个相关结果"
        ))
    
    return {
        "search_results": result.get("search_results", {}),
        "crawled_contents": result.get("crawled_contents", []),
        "messages": [AIMessage(content=f"搜索完成，找到 {len(result.get('crawled_contents', []))} 个相关网页")],
        "next_step": "summarize"
    }


async def summarize_node(state: AgentState) -> Dict[str, Any]:
    """
    汇总节点：生成网络搜索结果的摘要
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="summarizing",
            title="📝 生成回答",
            description="正在整合信息并生成回答..."
        ))
    
    crawl_flag = state.get("crawled_flag")
    crawl_content=state.get("crawled_contents", [])

    search_res=state.get("search_results", [])
    # 若没有配置 crawl 就用search的内容
    search_summary=prompts.format_crawled_content(crawl_content) if crawl_flag else prompts.format_search_content(search_res)
    query = state["search_query"]
    conversation_history = state.get("conversation_history", [])
    
    # 格式化内容
    history_context = prompts.format_history_context(conversation_history)
    # content_text = prompts.format_crawled_content(search_summary)
    
    # 构建提示词
    prompt = prompts.WEB_SEARCH_SUMMARY.format(
        history_context=history_context,
        query=query,
        content_text=search_summary
    )
    
    # 流式生成
    llm_service = get_llm_service()
    
    async def status_callback_wrapper(event_type: str, data: Dict[str, Any]):
        if callback:
            await callback.emit(event_type, data)
    
    content, _ = await llm_service.generate(
        prompt=prompt,
        streaming=True,
        callback=status_callback_wrapper
    )
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="summarize_complete",
            title="✓ 回答完成",
            description="已生成完整回答"
        ))
    
    return {
        "final_summary": content,
        "messages": [AIMessage(content=content)],
        "next_step": "end"
    }


async def doc_search_node(state: AgentState) -> Dict[str, Any]:
    """
    文档搜索节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    query = state["search_query"]
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="docqa_search",
            title="📚 搜索文档库",
            description=f"正在文档库中搜索: {query}"
        ))
    
    # 调用 RAG 服务
    rag_service = get_rag_service()
    try:
        result = await rag_service.search_and_format(query)
        
        if callback:
            await callback.emit(EventType.STEP, StepEvent.create(
                step="docqa_found",
                title="✓ 文档检索完成",
                description=result.get("description", "已完成文档内容检索") 
            ))
        
        return {
            "docqa_content": result.get("formatted_content", ""),
            "messages": [AIMessage(content="已从文档库检索到相关内容")],
            "description": result.get("description", ""),
            "next_step": "llm_node"
        }
    except Exception as e:
        logger.error(f"文档检索失败: {e}")
        if callback:
            await callback.emit(EventType.ERROR, {
                "step": "docqa_error",
                "message": f"文档检索错误: {str(e)}"
            })
        return {
            "docqa_content": "文档检索失败",
            "messages": [AIMessage(content="文档检索失败")],
            "next_step": "llm_node"
        }


async def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    文档问答的汇总回答节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="generating",
            title="✍️ 生成回答",
            description="基于文档内容生成回答..."
        ))
    
    docqa_content = state.get("docqa_content", "")
    query = state["search_query"]
    conversation_history = state.get("conversation_history", [])
    
    # 格式化上下文
    history_context = prompts.format_history_context(conversation_history)
    
    # 构建提示词
    prompt = prompts.DOC_QA.format(
        history_context=history_context,
        query=query,
        docqa_content=docqa_content
    )
    
    # 流式生成
    llm_service = get_llm_service()
    
    async def status_callback_wrapper(event_type: str, data: Dict[str, Any]):
        if callback:
            await callback.emit(event_type, data)
    
    content, _ = await llm_service.generate(
        prompt=prompt,
        streaming=True,
        callback=status_callback_wrapper
    )
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="generate_complete",
            title="✓ 回答完成",
            description="已基于文档生成完整回答"
        ))
    
    return {
        "final_summary": content,
        "messages": [AIMessage(content=content)],
        "next_step": "end"
    }


async def chat_node(state: AgentState) -> Dict[str, Any]:
    """
    聊天对话节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    callback = state.get("status_callback")
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="chatting",
            title="💬 对话中",
            description="正在生成回复..."
        ))
    
    query = state["search_query"]
    conversation_history = [{'role':"system","content":"用户偏好："+state.get('user_profile') if state.get('user_profile') else "用户无任何偏好 正常回答"}]+state.get("conversation_history", [])
    
    # 格式化上下文
    history_context = prompts.format_history_context(conversation_history)
    
    # 构建提示词
    prompt = prompts.CHAT.format(
        history_context=history_context,
        query=query
    )
    
    # 流式生成
    llm_service = get_llm_service()
    
    async def status_callback_wrapper(event_type: str, data: Dict[str, Any]):
        if callback:
            await callback.emit(event_type, data)
    
    content, _ = await llm_service.generate(
        prompt=prompt,
        streaming=True,
        callback=status_callback_wrapper
    )
    
    if callback:
        await callback.emit(EventType.STEP, StepEvent.create(
            step="chat_complete",
            title="✓ 回复完成",
            description="已生成回复"
        ))
    
    return {
        "final_summary": content,
        "messages": [AIMessage(content=content)],
        "next_step": "end"
    }