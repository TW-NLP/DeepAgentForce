import asyncio
import json
import logging
import requests

# 模拟日志对象，防止代码报错
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGService:
    """RAG 文档检索服务"""
    
    def __init__(self):
        """初始化 RAG 服务"""
        # 构建搜索 URL
        self.search_url = f"http://localhost:8000/api/rag/query"
          
    async def search_documents(self, query: str):
        logger.info(f"🔍 RAG 服务执行文档搜索: {query}")
        
        # 构建搜索请求
        payload = {
            "question": query
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 发送请求 (注意：requests 是同步库，在 async 中建议后续改为 httpx)
        try:
            response = requests.post(
                self.search_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            response.raise_for_status()
            
            # 解析结果
            results = response.json()
            logger.info(f"✅ RAG 搜索完成，结果如下：{json.dumps(results, ensure_ascii=False)}")
            return [results]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 请求失败: {e}")
            return []

# --- 测试调用部分 ---
async def main():
    # 1. 实例化服务
    rag_service = RAGService()
    
    # 2. 准备测试问题
    test_query = "大模型技术"
    
    print(f"\n--- 开始测试 RAG 服务 ---")
    
    # 3. 调用异步方法
    try:
        results = await rag_service.search_documents(test_query)
        
        # 4. 打印最终返回结果
        print("\n[最终返回结果]:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    # 启动异步任务
    asyncio.run(main())