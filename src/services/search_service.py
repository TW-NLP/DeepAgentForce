"""
搜索服务模块
封装 Tavily 和 Firecrawl 的调用
"""

import json
import logging
from typing import Optional, List, Dict, Any
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from config.settings import settings

logger = logging.getLogger(__name__)


class SearchService:
    """搜索服务类"""
    
    def __init__(self):
        """初始化搜索服务"""
        self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.firecrawl_client = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY) if settings.FIRECRAWL_API_KEY else None
        
        logger.info("✅ 搜索服务初始化成功")
    
    async def web_search(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            搜索结果字典
        """
        try:
            
            logger.info(f"🔍 执行网络搜索: {query}")
            
            response = self.tavily_client.search(
                query=query
            )
            
            logger.info(f"✅ 搜索完成，找到 {len(response.get('results', []))} 个结果，内容如下：{json.dumps(response, ensure_ascii=False)}")
            return response
            
        except Exception as e:
            logger.error(f"❌ 网络搜索失败: {e}")
            raise
    
    async def crawl_url(
        self,
        url: str,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        爬取单个网页
        
        Args:
            url: 网页 URL
            max_length: 内容最大长度
            
        Returns:
            爬取结果，包含 markdown 格式的内容
        """
        try:
            if not self.firecrawl_client:
                return ''
            logger.info(f"🕷️ 爬取网页: {url}")
            
            result = self.firecrawl_client.scrape(url)
            
            # 限制内容长度
            max_len = max_length or settings.FIRECRAWL_MAX_CONTENT_LENGTH
            if hasattr(result, 'markdown') and result.markdown:
                result.markdown = result.markdown[:max_len]
            
            logger.info(f"✅ 爬取成功: {url}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 爬取失败 {url}: {e}")
            raise
    
    async def crawl_multiple_urls(
        self,
        urls: List[str],
        max_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        并行爬取多个网页
        
        Args:
            urls: URL 列表
            max_length: 内容最大长度
            
        Returns:
            爬取结果列表
        """
        results = []
        
        for url in urls[:settings.MAX_URLS_TO_CRAWL]:
            try:
                result = await self.crawl_url(url, max_length)
                results.append({
                    "url": url,
                    "success": True,
                    "content": result
                })
            except Exception as e:
                logger.warning(f"跳过爬取失败的 URL: {url}")
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"📊 批量爬取完成: {success_count}/{len(urls)} 成功，内容如下：{results}")
        
        return results
    
    async def search_and_crawl(
        self,
        query: str,
        num_urls: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        一站式搜索+爬取
        
        Args:
            query: 搜索查询
            num_urls: 要爬取的 URL 数量
            
        Returns:
            包含搜索结果和爬取内容的字典
        """
        # 执行搜索
        search_results = await self.web_search(query)
        
        # 提取 URL
        urls = [
            result['url'] 
            for result in search_results.get('results', [])
        ][:num_urls or settings.MAX_URLS_TO_CRAWL]
        
        # 爬取内容
        crawled_results = await self.crawl_multiple_urls(urls)
        logger.info(f"搜索的全部内容如下：{
            {
            "query": query,
            "search_results": search_results,
            "crawled_flag": True if self.firecrawl_client else False,
            "crawled_contents": [
                {
                    "url": r["url"],
                    "title": next(
                        (res['title'] for res in search_results.get('results', []) 
                         if res['url'] == r["url"]),
                        r["url"]
                    ),
                    "snippet": next(
                        (res['content'] for res in search_results.get('results', []) 
                         if res['url'] == r["url"]),
                        ""
                    ),
                    "full_content": r["content"].markdown[:settings.FIRECRAWL_MAX_CONTENT_LENGTH]
                    if r.get("success") and hasattr(r["content"], 'markdown')
                    else ""
                }
                for r in crawled_results
                if r.get("success")
            ]
        }
        }")
        # 组合结果
        return {
            "query": query,
            "search_results": search_results,
            "crawled_contents": [
                {
                    "url": r["url"],
                    "title": next(
                        (res['title'] for res in search_results.get('results', []) 
                         if res['url'] == r["url"]),
                        r["url"]
                    ),
                    "snippet": next(
                        (res['content'] for res in search_results.get('results', []) 
                         if res['url'] == r["url"]),
                        ""
                    ),
                    "full_content": r["content"].markdown[:settings.FIRECRAWL_MAX_CONTENT_LENGTH]
                    if r.get("success") and hasattr(r["content"], 'markdown')
                    else ""
                }
                for r in crawled_results
                if r.get("success")
            ]
        }


# 创建全局搜索服务实例
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """获取搜索服务单例"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service