"""
提示词模板管理
集中管理所有的提示词模板
"""


class PromptTemplates:
    """提示词模板类"""
    
    # ==================== 任务规划提示词 ====================
    TASK_PLANNING = """
{history_context}

当前任务: {query}
用户偏好: {user_profile}
请分析这个任务需要哪些步骤,并以 JSON 格式返回执行计划:

{{
  "task_type": "simple|composite",
  "steps": [
    {{
      "action": "web_search|doc_search|chat",
      "reason": "为什么需要这一步",
      "query": "具体的搜索查询"
    }}
  ],
  "final_action": "synthesize|direct_answer"
}}

判断标准:
- **web_search**: 需要外部信息(新闻、产品、技术、公司信息、开源项目等)
- **doc_search**: 需要公司内部资料(制度、流程、现状、数据等)
- **chat**: 简单对话或不需要搜索
- **composite**: 需要多个数据源(如对比分析、结合内外部信息)

示例1:
输入: "对比 A 公司与 B 公司的新产品并结合我司现状给出建议"
输出:
{{
  "task_type": "composite",
  "steps": [
    {{"action": "web_search", "reason": "获取 A 公司产品信息", "query": "A公司最新产品特性"}},
    {{"action": "web_search", "reason": "获取 B 公司产品信息", "query": "B公司最新产品特性"}},
    {{"action": "doc_search", "reason": "了解我司现状", "query": "公司产品策略 业务现状"}}
  ],
  "final_action": "synthesize"
}}

示例2:
输入: "Python最新版本有哪些新特性?"
输出:
{{
  "task_type": "simple",
  "steps": [
    {{"action": "web_search", "reason": "查找Python最新版本信息", "query": "Python最新版本新特性"}}
  ],
  "final_action": "direct_answer"
}}

示例3:
输入: "你好"
输出:
{{
  "task_type": "simple",
  "steps": [
    {{"action": "chat", "reason": "简单问候", "query": "你好"}}
  ],
  "final_action": "direct_answer"
}}

示例4:
输入: "公司的请假流程是什么?"
输出:
{{
  "task_type": "simple",
  "steps": [
    {{"action": "doc_search", "reason": "查询公司内部制度", "query": "请假流程 审批制度"}}
  ],
  "final_action": "direct_answer"
}}

只返回 JSON,不要其他内容。
"""

    # ==================== 综合分析提示词 ====================
    SYNTHESIS = """
{history_context}

**用户任务**: {query}

**收集到的信息**:

{web_context}

{doc_context}

**请完成以下任务**:
基于以上信息，给出一个清晰、专业的综合建议，直接回答用户的问题。如果内部信息和问题不相关，就直接根据外部信息进行回答，要求引用相关来源支持观点，保持回答简洁有力。
"""

    # ==================== 网络搜索总结提示词 ====================
    WEB_SEARCH_SUMMARY = """
{history_context}

用户问题: {query}

基于以下网络搜索结果，请提供一个清晰、准确的回答：

{content_text}

要求：
1. 直接回答用户的问题
2. 引用相关来源支持观点
3. 如果信息不足，明确说明
4. 保持回答简洁专业
"""

    # ==================== 文档问答提示词 ====================
    DOC_QA = """
{history_context}

用户问题: {query}

基于以下公司文档内容，请提供准确的回答：

{docqa_content}

要求：
1. 基于文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说明
3. 引用具体的文档内容
4. 保持回答清晰专业
"""

    # ==================== 聊天对话提示词 ====================
    CHAT = """
{history_context}

用户: {query}

请自然地回复用户的消息。保持友好、专业的态度。
"""

    # ==================== 上下文格式化 ====================
    @staticmethod
    def format_history_context(conversation_history: list, limit: int = 10) -> str:
        """格式化对话历史"""
        if not conversation_history:
            return ""
        
        history_context = "\n对话历史:\n"
        for msg in conversation_history[-limit:]:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:300]  # 限制长度
            history_context += f"{role}: {content}\n"
        
        return history_context
    
    @staticmethod
    def format_web_context(web_results: list) -> str:
        """格式化网络搜索结果"""
        if not web_results:
            return ""
        
        web_context = ""
        for i, item in enumerate(web_results, 1):
            web_context += f"\n**外部信息 {i}** ({item.get('reason', '无来源说明')}):\n"
            
            # 检查 item['content'] 是否存在且为列表
            content_list = item.get('content', [])
            if not isinstance(content_list, list):
                continue

            for doc in content_list:
                #优先获取 snippet，如果为空则获取 full_content，最后兜底为空字符串
                text_content = doc.get('snippet') or doc.get('full_content') or ""
                
                # 去除空白字符并截断
                if text_content:
                    web_context += f"- {text_content.strip()[:500]}\n"
        
        return web_context
    
    @staticmethod
    def format_doc_context(doc_results: list) -> str:
        """格式化文档搜索结果"""
        if not doc_results:
            return ""
        
        doc_context = ""
        for i, item in enumerate(doc_results, 1):
            raw_content = item.get('content', "")
            
            final_text = ""
            if isinstance(raw_content, dict):
                # 核心内容在 'answer' 字段中
                final_text = raw_content.get('answer', str(raw_content))
            elif isinstance(raw_content, str):
                final_text = raw_content
            else:
                final_text = str(raw_content)
            
            # 确保是字符串后再切片
            doc_context += f"\n**内部资料 {i}** ({item.get('reason', '无说明')}):\n{str(final_text)[:800]}\n"
        
        return doc_context
    
    @staticmethod
    def format_crawled_content(crawled_contents: list) -> str:
        """格式化爬取的网页内容"""
        if not crawled_contents:
            return ""
        
        content_text = "\n\n".join([
            f"【来源 {i+1}】{item['title']}\n{item['full_content'][:800]}"
            for i, item in enumerate(crawled_contents)
        ])
        
        return content_text

    @staticmethod
    def format_search_content(search_results: dict) -> str:
        """格式化爬取的网页内容"""
        if not search_results.get("results"):
            return ""
        
        content_text = "\n\n".join([
            f"【网址 \n {item['url']}\n来源 {i+1}】{item['title']}\n{item['content'][:800]}"
            for i, item in enumerate(search_results.get("results", []))
        ])
        
        return content_text


# 导出提示词模板实例
prompts = PromptTemplates()