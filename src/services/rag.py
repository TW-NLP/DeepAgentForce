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
    多租户支持：每个租户有独立的 collection 和元数据存储
    - 文档元数据 -> 本地 JSON (按租户)
    - 文本块 & 向量 -> Milvus (按租户 collection)
    """
    
    def __init__(self, settings):
        """
        初始化 Milvus 客户端和 OpenAI 客户端"""
        self.settings = settings
        logger.info(f"📌 RAG 初始化 - EMBEDDING_URL: {self.settings.EMBEDDING_URL}")
        self.client_rag = AsyncOpenAI(api_key=self.settings.EMBEDDING_API_KEY, base_url=self.settings.EMBEDDING_URL)
        self.client_llm = AsyncOpenAI(api_key=self.settings.LLM_API_KEY, base_url=self.settings.LLM_URL)
        
        # Milvus Client
        self.milvus = MilvusClient(uri=self.settings.MILVUS_URL)
        # 🆕 多租户：维护每个租户的元数据（使用 tenant_uuid）
        self._tenant_documents: dict[str, Dict] = {}  # {tenant_uuid: {doc_uuid: dict}}


    def _get_tenant_collection(self, tenant_uuid: Optional[str] = None) -> str:
        """获取租户专属的 collection 名称（使用 tenant_uuid）"""
        base = self.settings.MILVUS_COLLECTION
        if tenant_uuid:
            # UUID 可能包含连字符，替换为下划线以符合 Milvus collection 命名规范
            safe_uuid = tenant_uuid.replace('-', '_')
            return f"{base}_{safe_uuid}"
        return f"{base}_default"


    def _init_tenant_collection(self, tenant_uuid: Optional[str] = None):
        """初始化租户专属的 Milvus 集合"""
        collection_name = self._get_tenant_collection(tenant_uuid)
        if not self.milvus.has_collection(collection_name):
            # 1. 创建集合
            self.milvus.create_collection(
                collection_name=collection_name,
                dimension=self.settings.EMBEDDING_DIM,
                metric_type="COSINE",
                auto_id=True, 
                enable_dynamic_field=True
            )
            
            # 2. 准备索引参数
            index_params = self.milvus.prepare_index_params()

            # 3. 为向量字段创建索引
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )

            # 4. 执行创建索引
            self.milvus.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            logger.info(f"✅ 创建租户 Milvus 集合: {collection_name}")


    def _get_tenant_meta_file(self, tenant_uuid: Optional[int] = None) -> Path:
        """获取租户专属的元数据文件路径"""
        if tenant_uuid is None:
            tenant_uuid = "default"
        return self.settings.MILVUS_DIR / f"doc_metadata_{tenant_uuid}.json"


    def _load_tenant_metadata(self, tenant_uuid: Optional[int] = None) -> Dict:
        """加载指定租户的元数据"""
        meta_file = self._get_tenant_meta_file(tenant_uuid)
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


    def _save_tenant_metadata(self, tenant_uuid: Optional[int] = None):
        """保存指定租户的元数据"""
        meta_file = self._get_tenant_meta_file(tenant_uuid)
        documents = self._tenant_documents.get(tenant_uuid or "default", {})
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)


    def _ensure_tenant_ready(self, tenant_uuid: Optional[int] = None):
        """确保租户的 collection 和元数据已初始化"""
        key = tenant_uuid or "default"
        if key not in self._tenant_documents:
            self._init_tenant_collection(tenant_uuid)
            self._tenant_documents[key] = self._load_tenant_metadata(tenant_uuid)


    def _get_tenant_documents(self, tenant_uuid: Optional[int] = None) -> Dict:
        """获取租户的文档元数据字典"""
        self._ensure_tenant_ready(tenant_uuid)
        key = tenant_uuid or "default"
        return self._tenant_documents.get(key, {})


    async def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """批量获取 Embedding"""
        if not texts: return []

        resp = await self.client_rag.embeddings.create(input=texts, model=self.settings.EMBEDDING_MODEL)
        return [d.embedding for d in resp.data]

    # --- 功能实现 ---

    async def add_document(self, file_path: str, metadata: Dict, tenant_uuid: Optional[int] = None) -> str:
        """
        解析 -> 分块 -> 向量化 -> 存入租户专属 Milvus
        
        Args:
            file_path: 文件路径
            metadata: 文档元数据
            tenant_uuid: 租户ID 🆕
        """
        doc_uuid = str(uuid.uuid4())
        self._ensure_tenant_ready(tenant_uuid)
        
        # 1. 解析与分块
        parser = DocumentParser()
        chunker = TextChunker()
        
        text = parser.parse_file(Path(file_path))
        chunks = chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("文档解析为空")

        logger.info(f"📄 处理文档 [{tenant_uuid}]: {metadata['title']} ({len(chunks)} chunks)")

        # 2. 向量化
        embeddings = await self._get_embedding(chunks)

        # 3. 构造 Milvus 数据
        milvus_data = []
        for chunk_text, vector in zip(chunks, embeddings):
            milvus_data.append({
                "vector": vector,
                "text": chunk_text,
                "doc_id": doc_uuid,
                "source": metadata['original_filename'],
                "tenant_uuid": tenant_uuid  # 🆕 记录租户
            })

        # 4. 写入租户专属 Milvus collection
        collection_name = self._get_tenant_collection(tenant_uuid)
        self.milvus.insert(collection_name=collection_name, data=milvus_data)

        # 5. 更新租户本地元数据
        key = tenant_uuid or "default"
        metadata.update({
            "uuid": doc_uuid,
            "path": file_path,
            "chunks_count": len(chunks),
            "added_at": datetime.now().isoformat()
        })
        self._tenant_documents[key][doc_uuid] = metadata
        self._save_tenant_metadata(tenant_uuid)

        return doc_uuid

    def remove_document(self, doc_id: str, tenant_uuid: Optional[int] = None):
        """
        从租户 Milvus 和本地元数据中删除
        
        Args:
            doc_id: 文档ID
            tenant_uuid: 租户ID 🆕
        """
        self._ensure_tenant_ready(tenant_uuid)
        key = tenant_uuid or "default"
        documents = self._tenant_documents.get(key, {})
        
        if doc_id not in documents:
            raise ValueError(f"文档 ID 不存在: {doc_id}")

        # 1. 从租户 Milvus 删除 (根据 doc_id 过滤，同时验证 tenant_uuid)
        collection_name = self._get_tenant_collection(tenant_uuid)
        delete_expr = f'doc_id == "{doc_id}"'
        self.milvus.delete(collection_name=collection_name, filter=delete_expr)
        
        # 2. 删除本地元数据
        doc_info = documents.pop(doc_id)
        self._save_tenant_metadata(tenant_uuid)
        
        # 3. 删除物理文件
        path = Path(doc_info['path'])
        if path.exists():
            path.unlink()

        logger.info(f"🗑️ 文档已删除: {doc_id} (tenant={tenant_uuid})")

    def list_documents(self, tenant_uuid: Optional[int] = None) -> List[Dict]:
        """
        列出指定租户的所有文档
        
        Args:
            tenant_uuid: 租户ID 🆕
        """
        self._ensure_tenant_ready(tenant_uuid)
        key = tenant_uuid or "default"
        documents = self._tenant_documents.get(key, {})
        return list(documents.values())

    def get_stats(self, tenant_uuid: Optional[int] = None):
        """获取租户的索引统计"""
        self._ensure_tenant_ready(tenant_uuid)
        key = tenant_uuid or "default"
        documents = self._tenant_documents.get(key, {})
        total_docs = len(documents)
        
        # Milvus 统计行数
        collection_name = self._get_tenant_collection(tenant_uuid)
        res = self.milvus.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"]
        )
        total_chunks = res[0]["count(*)"] if res else 0
        return {
            "index_status": "Indexed" if total_chunks > 0 else "Empty",
            "total_documents": total_docs,
            "total_chunks": total_chunks
        }

    async def query(self, question: str, top_k: int = 5, tenant_uuid: Optional[int] = None) -> str:
        """
        向量检索 + 返回上下文
        
        Args:
            question: 查询问题
            top_k: 返回的 chunk 数量
            tenant_uuid: 租户ID 🆕
        """
        self._ensure_tenant_ready(tenant_uuid)
        
        # 1. Embedding
        q_vec = (await self._get_embedding([question]))[0]

        # 2. Milvus Search (租户专属 collection)
        collection_name = self._get_tenant_collection(tenant_uuid)
        results = self.milvus.search(
            collection_name=collection_name,
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
