import os
import json
import shutil
import logging
import hashlib
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, APIRouter, Body
from pydantic import BaseModel
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from pymilvus import MilvusClient, DataType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MilvusRAG")

# ==================== Pydantic Models (接口定义) ====================

class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    document_name: str
    chunks_count: int
    uploaded_at: str

class DocumentInfo(BaseModel):
    document_id: str
    name: str
    path: str
    chunks: int
    uploaded_at: str
    metadata: Dict

class ListDocumentsResponse(BaseModel):
    success: bool
    total: int
    documents: List[DocumentInfo]

class IndexStatusResponse(BaseModel):
    success: bool
    is_indexed: bool
    total_documents: int
    total_chunks: int
    message: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5 # 对应原接口的 top_k_communities，基础RAG里即为 top_k chunks

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    processing_time: float

class DeleteResponse(BaseModel):
    success: bool
    message: str
    document_id: str

# ==================== 基础工具类 ====================

class DocumentParser:
    """文档解析器"""
    @staticmethod
    def parse_file(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        try:
            if ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join([p.extract_text() or "" for p in pdf.pages])
            elif ext in ['.docx', '.doc']:
                doc = DocxDocument(file_path)
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string(index=False)
            elif ext in ['.md', '.markdown']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            logger.error(f"解析失败 {file_path}: {e}")
            raise

class TextChunker:
    """分块器"""
    def __init__(self, chunk_size=600, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunks.append(self.encoding.decode(chunk_tokens))
        return chunks


class MilvusRAGPipeline():
    """
    基础 RAG Pipeline
    - 文档元数据 -> 本地 JSON
    - 文本块 & 向量 -> Milvus
    """
    
    def __init__(self,settings):
        """
        初始化 Milvus 客户端和 OpenAI 客户端"""
        self.settings=settings

        self.client_rag = AsyncOpenAI(api_key=self.settings.EMBEDDING_API_KEY, base_url=self.settings.EMBEDDING_URL)

        self.client_llm = AsyncOpenAI(api_key=self.settings.LLM_API_KEY, base_url=self.settings.LLM_URL)
        
        # Milvus Client
        self.milvus = MilvusClient(uri=self.settings.MILVUS_URL)
        self._init_collection()
        
        # 本地元数据存储 (用于快速列表)
        self.meta_file = self.settings.MILVUS_DIR / "doc_metadata.json"
        self.documents = self._load_local_metadata() # {doc_uuid: dict}


    def _init_collection(self):
        """初始化 Milvus 集合"""
        if not self.milvus.has_collection(self.settings.MILVUS_COLLECTION):
            # 1. 创建集合 (简易模式会自动创建名为 "vector" 的向量字段)
            self.milvus.create_collection(
                collection_name=self.settings.MILVUS_COLLECTION,
                dimension=self.settings.EMBEDDING_DIM,
                metric_type="COSINE",
                auto_id=True, 
                enable_dynamic_field=True
            )
            
            # 2. 准备索引参数
            index_params = self.milvus.prepare_index_params()

            # 3. 必须先为向量字段 "vector" 创建索引 (这是简易模式的默认字段名)
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX", # 自动选择最合适的索引
                metric_type="COSINE"
            )

            # 4. 为 doc_id 创建标量索引 (加速删除和过滤)
            # 注意：Milvus 中字符串通常使用 "STL_SORT" 或默认索引类型，"Trie" 在某些版本有特定限制
            # index_params.add_index(
            #     field_name="doc_id",
            #     index_type="INVERTED"  # 推荐使用倒排索引，加速精确匹配
            # )

            # 5. 执行创建索引
            self.milvus.create_index(
                collection_name=self.settings.MILVUS_COLLECTION,
                index_params=index_params
            )
            
            logger.info(f"✅ 创建 Milvus 集合及索引: {self.settings.MILVUS_COLLECTION}")
    def _load_local_metadata(self) -> Dict:
        if self.meta_file.exists():
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_local_metadata(self):
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    async def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """批量获取 Embedding"""
        if not texts: return []

        resp = await self.client_rag.embeddings.create(input=texts, model=self.settings.EMBEDDING_MODEL)
        return [d.embedding for d in resp.data]

    # --- 功能实现 ---

    async def add_document(self, file_path: str, metadata: Dict) -> str:
        """解析 -> 分块 -> 向量化 -> 存入 Milvus"""
        doc_uuid = str(uuid.uuid4())
        
        # 1. 解析与分块
        parser = DocumentParser()
        chunker = TextChunker()
        
        text = parser.parse_file(Path(file_path))
        chunks = chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("文档解析为空")

        logger.info(f"📄 处理文档: {metadata['title']} ({len(chunks)} chunks)")

        # 2. 向量化
        embeddings = await self._get_embedding(chunks)

        # 3. 构造 Milvus 数据
        milvus_data = []
        for chunk_text, vector in zip(chunks, embeddings):
            milvus_data.append({
                "vector": vector,
                "text": chunk_text,
                "doc_id": doc_uuid,
                "source": metadata['original_filename']
            })

        # 4. 写入 Milvus
        self.milvus.insert(collection_name=self.settings.MILVUS_COLLECTION, data=milvus_data)

        # 5. 更新本地元数据
        metadata.update({
            "uuid": doc_uuid,
            "path": file_path,
            "chunks_count": len(chunks),
            "added_at": datetime.now().isoformat()
        })
        self.documents[doc_uuid] = metadata
        self._save_local_metadata()

        return doc_uuid

    def remove_document(self, doc_id: str):
        """从 Milvus 和本地元数据中删除"""
        if doc_id not in self.documents:
            raise ValueError(f"文档 ID 不存在: {doc_id}")

        # 1. 从 Milvus 删除 (根据 doc_id 过滤)
        delete_expr = f'doc_id == "{doc_id}"'
        self.milvus.delete(collection_name=self.settings.MILVUS_COLLECTION, filter=delete_expr)
        
        # 2. 删除本地元数据
        doc_info = self.documents.pop(doc_id)
        self._save_local_metadata()
        
        # 3. 删除物理文件 (可选)
        path = Path(doc_info['path'])
        if path.exists():
            path.unlink()

        logger.info(f"🗑️ 文档已删除: {doc_id}")

    def list_documents(self) -> List[Dict]:
        return list(self.documents.values())

    def get_stats(self):
        total_docs = len(self.documents)
        # Milvus 统计行数估算
        res = self.milvus.query(collection_name=self.settings.MILVUS_COLLECTION, filter="", output_fields=["count(*)"])
        total_chunks = res[0]["count(*)"] if res else 0
        return {
            "index_status": "Indexed" if total_chunks > 0 else "Empty",
            "total_documents": total_docs,
            "total_chunks": total_chunks
        }

    async def query(self, question: str, top_k: int = 5) -> str:
        """向量检索 + LLM 生成"""
        # 1. Embedding
        q_vec = (await self._get_embedding([question]))[0]

        # 2. Milvus Search
        results = self.milvus.search(
            collection_name=self.settings.MILVUS_COLLECTION,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "source"]
        )

        if not results or not results[0]:
            return "未找到相关文档信息。"

        # 3. 构造 Context
        retrieved_texts = [hit['entity']['text'] for hit in results[0]]
        context_str = "\n\n---\n\n".join(retrieved_texts)

        return context_str

        # # 4. LLM Generate
        # system_prompt = "你是一个智能助手。请根据提供的上下文信息回答用户的问题。如果上下文中没有答案，请诚实告知。"
        # user_prompt = f"上下文:\n{context_str}\n\n问题: {question}"

        # resp = await self.client_llm.chat.completions.create(
        #     model=self.settings.LLM_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # return resp.choices[0].message.content




    def save_upload_file(self, file: UploadFile) -> Path:
        file_path = self.settings.UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path
