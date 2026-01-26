import httpx
import json
import logging
import requests
from typing import Optional, List, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 文档检索服务"""
    
    def __init__(self):
        """初始化 RAG 服务"""
        # 构建搜索 URL
        self.search_url = settings.RAG_URL
          
    async def search_documents(self, query: str) -> List[Dict]:
        """使用 httpx 进行异步非阻塞调用，解决超时卡死问题"""
        payload = {"question": query}
        
        async with httpx.AsyncClient() as client:
            try:
                # 设置更长的超时时间，例如 30秒 或 60秒
                response = await client.post(
                    self.search_url, 
                    json=payload, 
                    timeout=30.0 
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ RAG服务返回结果: {result}")
                # 假设返回的是一个对象，我们把它放进列表统一处理
                return result
            except httpx.TimeoutException:
                logger.error(f"❌ RAG服务请求超时 (Query: {query})")
                return ''
            except Exception as e:
                logger.error(f"❌ RAG服务请求异常: {e}")
                return ''
    
        
    async def format_search_results(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 800
    ) -> str:
        """
        格式化搜索结果
        
        Args:
            documents: 文档列表
            max_length: 每个文档的最大长度
            
        Returns:
            格式化后的文档内容
        """
        if not documents:
            return "未找到相关文档"
        
        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')[:max_length]
            score = doc.get('score', 0)
            if score:
            
                formatted_parts.append(
                    f"文档片段 {i} (相关度: {score:.2f}):\n{content}"
                )
            else:
                formatted_parts.append(
                    f"文档片段 {i}:\n{content}"
                )
        
        return '\n\n'.join(formatted_parts)
    
    async def search_and_format(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        搜索并格式化文档
        
        Args:
            query: 搜索查询
            
        Returns:
            包含原始文档和格式化内容的字典
        """
        formatted_content = await self.search_documents(query)
        logger.info(f"✅ RAG服务返回结果: {formatted_content}")
        
        return {
            "query": query,
            "formatted_content": formatted_content,
            "description":"已经为您检索到相关知识库内容。"
        }
    
    def health_check(self) -> bool:
        """
        检查 Elasticsearch 服务健康状态
        
        Returns:
            是否健康
        """
        try:
            response = requests.get(
                f"{self.host}/_cluster/health",
                auth=self.auth,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ RAG 服务健康检查失败: {e}")
            return False


# 创建全局 RAG 服务实例
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service