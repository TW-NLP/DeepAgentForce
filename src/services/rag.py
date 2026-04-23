import os
import re
import json
import shutil
import logging
import hashlib
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import aiohttp
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, APIRouter, Body
from pydantic import BaseModel
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
import tiktoken
from openai import AsyncOpenAI
try:
    from pymilvus import MilvusClient, DataType
except Exception:  # pragma: no cover - optional backend
    MilvusClient = None
    DataType = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MilvusRAG")

# ==================== BM25 实现（纯 Python，无额外依赖）====================

class BM25:
    """轻量级 BM25 检索器（用于关键词召回）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs: Dict[int, Dict[str, int]] = {}
        self.doc_lens: List[int] = []
        self.doc_texts: List[str] = []
        self.idf: Dict[str, float] = {}
        self.vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text.lower())
        return [t for t in text.split() if t.strip()]

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.doc_texts = corpus
        self.doc_lens = []
        self.doc_freqs = {}
        self.vocab = {}

        for idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            freq = {}
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                tid = self.vocab[token]
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs[idx] = freq

        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0

        df = defaultdict(int)
        for freq_map in self.doc_freqs.values():
            for token in freq_map:
                df[token] += 1

        for token, freq in df.items():
            self.idf[token] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        doc_freqs = self.doc_freqs.get(doc_idx, {})
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        for token in tokens:
            if token not in self.vocab:
                continue
            tf = doc_freqs.get(token, 0)
            idf = self.idf.get(token, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-9))
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        scores = [(idx, self.score(query, idx)) for idx in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

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
    top_k: int = 5
    enable_rerank: bool = False
    enable_query_rewrite: bool = False


class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    processing_time: float
    stage_info: Optional[Dict[str, Any]] = None  # 记录各阶段耗时和状态

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
    多阶段增强 RAG Pipeline
    多租户支持：每个租户有独立的 collection 和元数据存储

    检索路线：
    ┌─────────────────────────────────────────────────────────────┐
    │  Stage 1: 双路召回（并行）                                  │
    │    ├─ 向量召回：Milvus ANN 检索                            │
    │    └─ 关键词召回：BM25（可选）                             │
    │  → Reciprocal Rank Fusion 融合                            │
    │                                                             │
    │  Stage 2: Rerank 重排（可选）                              │
    │    └─ Cross-Encoder 重排，输出 top_k                      │
    │                                                             │
    │  Stage 3: Query Rewrite 多路投票（可选）                   │
    │    ├─ LLM 生成 N 个子问题                                  │
    │    ├─ 并行向量检索每个子问题                                │
    │    └─ 按 RRF 融合多路结果                                  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, settings):
        self.settings = settings
        logger.info(f"📌 RAG 初始化 - EMBEDDING_URL: {self.settings.EMBEDDING_URL}")
        self._last_query_rewrite_info: List[Dict[str, Any]] = []
        self.client_rag = None
        if self.settings.EMBEDDING_URL and self.settings.EMBEDDING_MODEL:
            self.client_rag = AsyncOpenAI(
                api_key=self.settings.EMBEDDING_API_KEY,
                base_url=self.settings.EMBEDDING_BASE_URL
            )
        self.client_llm = None
        if self.settings.LLM_URL and self.settings.LLM_MODEL:
            self.client_llm = AsyncOpenAI(
                api_key=self.settings.LLM_API_KEY,
                base_url=self.settings.LLM_BASE_URL
            )
        self.milvus = None
        if MilvusClient is not None:
            try:
                self.milvus = MilvusClient(uri=self.settings.MILVUS_URL)
            except Exception as exc:
                logger.warning(f"Milvus 初始化失败，切换为轻量模式: {exc}")

        # 多租户：维护每个租户的元数据
        self._tenant_documents: dict[str, Dict] = {}
        # 多租户：BM25 索引缓存 {tenant_uuid: BM25}
        self._tenant_bm25: dict[str, BM25] = {}
        # 多租户：全量文本缓存 {tenant_uuid: List[str]}
        self._tenant_texts: dict[str, List[str]] = {}
        # 多租户：与文本缓存对齐的稳定 chunk 标识
        self._tenant_chunk_ids: dict[str, List[str]] = {}
        self._last_query_rewrite_info: List[Dict[str, Any]] = []


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
        if self.milvus is None:
            return
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

    def _build_chunk_uid(self, doc_uuid: str, chunk_index: int) -> str:
        return f"{doc_uuid}:{chunk_index}"

    def _get_hit_key(self, hit: Dict[str, Any]) -> Any:
        """返回用于 RRF / 投票的稳定 key。"""
        return hit.get("chunk_uid") or hit.get("doc_id") or hit.get("chunk_idx")


    async def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """批量获取 Embedding"""
        if not texts: return []
        if self.client_rag is None:
            return []

        resp = await self.client_rag.embeddings.create(input=texts, model=self.settings.EMBEDDING_MODEL)
        return [d.embedding for d in resp.data]

    # --- 功能实现 ---

    async def add_document(self, file_path: str, metadata: Dict, tenant_uuid: Optional[int] = None) -> str:
        doc_uuid = str(uuid.uuid4())
        self._ensure_tenant_ready(tenant_uuid)

        parser = DocumentParser()
        chunker = TextChunker()

        text = parser.parse_file(Path(file_path))
        chunks = chunker.chunk_text(text)

        if not chunks:
            raise ValueError("文档解析为空")

        logger.info(f"📄 处理文档 [{tenant_uuid}]: {metadata['title']} ({len(chunks)} chunks)")

        if self.milvus is not None and self.client_rag is not None and self.settings.EMBEDDING_MODEL:
            try:
                embeddings = await self._get_embedding(chunks)
                milvus_data = []
                chunk_uids = [self._build_chunk_uid(doc_uuid, idx) for idx in range(len(chunks))]
                for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                    milvus_data.append({
                        "vector": vector,
                        "text": chunk_text,
                        "doc_id": doc_uuid,
                        "chunk_uid": chunk_uids[idx],
                        "chunk_index": idx,
                        "source": metadata['original_filename'],
                        "tenant_uuid": tenant_uuid
                    })

                collection_name = self._get_tenant_collection(tenant_uuid)
                if milvus_data:
                    self.milvus.insert(collection_name=collection_name, data=milvus_data)
            except Exception as exc:
                logger.warning(f"向量索引写入失败，已降级为轻量模式: {exc}")

        key = tenant_uuid or "default"
        metadata.update({
            "uuid": doc_uuid,
            "path": file_path,
            "chunks_count": len(chunks),
            "added_at": datetime.now().isoformat()
        })
        metadata["chunks"] = chunks
        metadata["chunk_uids"] = [self._build_chunk_uid(doc_uuid, idx) for idx in range(len(chunks))]
        self._tenant_documents[key][doc_uuid] = metadata
        self._save_tenant_metadata(tenant_uuid)

        await self._rebuild_bm25(tenant_uuid)

        return doc_uuid

    def remove_document(self, doc_id: str, tenant_uuid: Optional[int] = None):
        self._ensure_tenant_ready(tenant_uuid)
        key = tenant_uuid or "default"
        documents = self._tenant_documents.get(key, {})

        if doc_id not in documents:
            raise ValueError(f"文档 ID 不存在: {doc_id}")

        if self.milvus is not None:
            collection_name = self._get_tenant_collection(tenant_uuid)
            delete_expr = f'doc_id == "{doc_id}"'
            self.milvus.delete(collection_name=collection_name, filter=delete_expr)

        doc_info = documents.pop(doc_id)
        self._save_tenant_metadata(tenant_uuid)

        path = Path(doc_info['path'])
        if path.exists():
            path.unlink()

        asyncio.create_task(self._rebuild_bm25(tenant_uuid))

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

        def _chunk_count(doc: Dict[str, Any]) -> int:
            chunks = doc.get("chunks", 0)
            if isinstance(chunks, list):
                return len(chunks)
            if isinstance(chunks, int):
                return chunks
            return int(doc.get("chunks_count", 0) or 0)

        total_chunks = sum(_chunk_count(doc) for doc in documents.values())
        if self.milvus is not None:
            try:
                collection_name = self._get_tenant_collection(tenant_uuid)
                res = self.milvus.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["count(*)"]
                )
                total_chunks = res[0]["count(*)"] if res else total_chunks
            except Exception as exc:
                logger.warning(f"Milvus 统计失败，使用本地元数据统计: {exc}")
        return {
            "index_status": "Indexed" if total_chunks > 0 else "Empty",
            "total_documents": total_docs,
            "total_chunks": total_chunks
        }

    # ==================== 多阶段检索核心 ====================

    async def _rebuild_bm25(self, tenant_uuid: Optional[int] = None):
        """从 Milvus 全量拉取文本，重建 BM25 索引"""
        key = tenant_uuid or "default"
        try:
            texts = []
            chunk_ids: List[str] = []
            if self.milvus is not None:
                collection_name = self._get_tenant_collection(tenant_uuid)
                res = self.milvus.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["text", "doc_id", "chunk_uid", "chunk_index"]
                )
                texts = [hit["text"] for hit in res]
                for hit in res:
                    chunk_uid = hit.get("chunk_uid")
                    if not chunk_uid:
                        doc_id = str(hit.get("doc_id", "unknown"))
                        chunk_index = int(hit.get("chunk_index", hit.get("id", 0)) or 0)
                        chunk_uid = self._build_chunk_uid(doc_id, chunk_index)
                    chunk_ids.append(str(chunk_uid))
            if not texts:
                for doc in self._tenant_documents.get(key, {}).values():
                    doc_uuid = str(doc.get("uuid", "unknown"))
                    doc_chunks = list(doc.get("chunks", []))
                    doc_chunk_uids = doc.get("chunk_uids") or [
                        self._build_chunk_uid(doc_uuid, idx) for idx in range(len(doc_chunks))
                    ]
                    texts.extend(doc_chunks)
                    chunk_ids.extend([str(uid) for uid in doc_chunk_uids])
            if texts:
                bm25 = BM25()
                bm25.fit(texts)
                self._tenant_bm25[key] = bm25
                self._tenant_texts[key] = texts
                self._tenant_chunk_ids[key] = chunk_ids
                logger.info(f"📦 BM25 索引重建完成 ({key}): {len(texts)} 条")
        except Exception as e:
            logger.warning(f"BM25 索引重建失败: {e}")

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Tuple[Any, float]]],
        k: int = 60
    ) -> List[Tuple[Any, float]]:
        """Reciprocal Rank Fusion — 融合多条有序召回结果"""
        scores: Dict[Any, float] = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, (chunk_idx, _score) in enumerate(ranked_list):
                scores[chunk_idx] += 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, s) for idx, s in sorted_items]

    def _has_vector_search(self) -> bool:
        return bool(self.client_rag and self.settings.EMBEDDING_MODEL)

    def _has_rerank_model(self) -> bool:
        return bool(self.settings.RERANK_API_URL and self.settings.RERANK_MODEL)

    def _has_query_rewrite_model(self) -> bool:
        return self.client_llm is not None

    def _get_keyword_top_k(self, top_k: int) -> int:
        return max(top_k, self.settings.KEYWORD_TOP_K, self.settings.RERANK_BATCH_SIZE)

    async def _retrieve_once(
        self,
        question: str,
        top_k: int,
        tenant_uuid: Optional[int] = None,
        *,
        allow_rerank: bool = False
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """执行一次完整检索：BM25 基线 + 可选向量召回 + 可选 Rerank。"""
        stage_info: Dict[str, Any] = {
            "bm25_enabled": True,
            "vector_enabled": self._has_vector_search(),
            "rerank_enabled": allow_rerank and self._has_rerank_model(),
        }

        bm25_top_k = self._get_keyword_top_k(top_k)
        bm25_hits = await self._stage1_bm25_search(question, bm25_top_k, tenant_uuid)
        stage_info["bm25_hits"] = len(bm25_hits)

        vector_hits: List[Dict] = []
        fused_hits: List[Dict] = bm25_hits
        if self._has_vector_search():
            vector_hits = await self._stage1_vector_search(question, top_k, tenant_uuid)
            stage_info["vector_hits"] = len(vector_hits)
            fused_hits = self._fuse_stage1(vector_hits, bm25_hits, tenant_uuid)
            stage_info["fusion_mode"] = "bm25+embedding+rrf"
            stage_info["fused_hits"] = len(fused_hits)
        else:
            stage_info["vector_hits"] = 0
            stage_info["fusion_mode"] = "bm25"

        if stage_info["rerank_enabled"] and fused_hits:
            rerank_batch = fused_hits[:self.settings.RERANK_BATCH_SIZE]
            reranked = await self._stage2_rerank(question, rerank_batch, tenant_uuid)
            stage_info["rerank_hits"] = len(reranked)
            stage_info["stage2"] = "rerank"
            final_hits = reranked[:top_k]
        else:
            stage_info["rerank_hits"] = 0
            stage_info["stage2"] = "none"
            final_hits = fused_hits[:top_k]

        stage_info["final_hits"] = len(final_hits)
        return final_hits, stage_info

    async def _stage1_vector_search(
        self,
        question: str,
        top_k: int,
        tenant_uuid: Optional[int] = None
    ) -> List[Dict]:
        """Stage 1a: 向量 ANN 召回"""
        if not self._has_vector_search() or self.milvus is None:
            return []
        q_vec = (await self._get_embedding([question]))[0]
        collection_name = self._get_tenant_collection(tenant_uuid)
        results = self.milvus.search(
            collection_name=collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "source", "doc_id", "chunk_uid", "chunk_index"]
        )
        hits = []
        for hit in (results[0] if results else []):
            entity = hit["entity"]
            chunk_uid = entity.get("chunk_uid")
            if not chunk_uid:
                doc_id = str(entity.get("doc_id", "unknown"))
                chunk_index = int(entity.get("chunk_index", hit.get("id", 0)) or 0)
                chunk_uid = self._build_chunk_uid(doc_id, chunk_index)
            hits.append({
                "chunk_idx": hit.get("id"),
                "chunk_uid": str(chunk_uid),
                "text": entity["text"],
                "source": entity.get("source", ""),
                "vector_score": hit.get("distance", 0)
            })
        return hits

    async def _stage1_bm25_search(
        self,
        question: str,
        top_k: int,
        tenant_uuid: Optional[int] = None
    ) -> List[Dict]:
        """Stage 1b: BM25 关键词召回"""
        key = tenant_uuid or "default"
        if key not in self._tenant_bm25:
            await self._rebuild_bm25(tenant_uuid)
        bm25 = self._tenant_bm25.get(key)
        texts = self._tenant_texts.get(key, [])
        if not texts:
            for doc in self._tenant_documents.get(key, {}).values():
                texts.extend(doc.get("chunks", []))
        if not bm25 or not texts:
            return []
        ranked = bm25.search(question, top_k)
        hits = []
        chunk_ids = self._tenant_chunk_ids.get(key, [])
        for chunk_idx, bm25_score in ranked:
            hits.append({
                "chunk_idx": chunk_idx,
                "chunk_uid": str(chunk_ids[chunk_idx]) if chunk_idx < len(chunk_ids) else str(chunk_idx),
                "text": texts[chunk_idx],
                "source": "",
                "bm25_score": bm25_score
            })
        return hits

    def _fuse_stage1(
        self,
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
        tenant_uuid: Optional[int] = None
    ) -> List[Dict]:
        """Stage 1 融合：对两条召回链路的 chunk_idx 做 RRF 融合"""
        key = tenant_uuid or "default"
        texts = self._tenant_texts.get(key, [])

        vector_ranked = [(self._get_hit_key(h), h["vector_score"]) for h in vector_hits]
        bm25_ranked = [(self._get_hit_key(h), h["bm25_score"]) for h in bm25_hits]

        if not vector_ranked and not bm25_ranked:
            return []
        if not bm25_ranked:
            return vector_hits
        if not vector_ranked:
            return bm25_hits

        fused_ranks = self._reciprocal_rank_fusion(
            [vector_ranked, bm25_ranked]
        )
        chunk_to_text = {self._get_hit_key(h): h["text"] for h in vector_hits + bm25_hits}
        chunk_to_source = {self._get_hit_key(h): h["source"] for h in vector_hits if "source" in h}

        fused_hits = []
        for chunk_idx, rrf_score in fused_ranks:
            text = chunk_to_text.get(chunk_idx, "")
            fused_hits.append({
                "chunk_idx": chunk_idx,
                "chunk_uid": chunk_idx,
                "text": text,
                "source": chunk_to_source.get(chunk_idx, ""),
                "rrf_score": rrf_score
            })
        return fused_hits

    async def _stage2_rerank(
        self,
        question: str,
        candidates: List[Dict],
        tenant_uuid: Optional[int] = None
    ) -> List[Dict]:
        """Stage 2: Cross-Encoder Rerank（对候选 chunk 重排）"""
        if not candidates:
            return []
        if not self._has_rerank_model():
            return candidates

        texts = [h["text"] for h in candidates]
        try:
            headers = {
                "Authorization": f"Bearer {self.settings.RERANK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.settings.RERANK_MODEL,
                "query": question,
                "documents": texts,
                "top_n": self.settings.RERANK_TOP_N,
                "return_documents": False
            }
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    self.settings.RERANK_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Rerank API 返回错误 {resp.status}")
                        return candidates[:self.settings.RERANK_TOP_N]
                    result = await resp.json()

            reranked = result.get("results", result.get("data", []))
            idx_score_map = {}
            for item in reranked:
                orig_idx = item.get("index", 0)
                score = item.get("relevance_score", item.get("score", 0))
                idx_score_map[orig_idx] = score

            for i, h in enumerate(candidates):
                h["rerank_score"] = idx_score_map.get(i, 0)

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:self.settings.RERANK_TOP_N]
        except Exception as e:
            logger.warning(f"Rerank 调用失败: {e}")
            return candidates[:self.settings.RERANK_TOP_N]

    async def _stage3_query_rewrite(
        self,
        question: str,
        top_k: int,
        tenant_uuid: Optional[int] = None,
        allow_rerank: bool = False
    ) -> List[Dict]:
        """Stage 3: Query Rewrite 多路投票
        1. LLM 生成多个子问题
        2. 每个子问题都走同一条检索链路（BM25 基线 + 可选向量 + 可选 Rerank）
        3. 对各子问题的最终排序结果做 RRF 投票
        """
        if not self._has_query_rewrite_model():
            hits, _ = await self._retrieve_once(
                question,
                top_k,
                tenant_uuid,
                allow_rerank=allow_rerank,
            )
            self._last_query_rewrite_info = []
            return hits

        n = self.settings.QUERY_REWRITE_NUM
        sub_questions = await self._generate_sub_questions(question, n)
        variants: List[str] = [question]
        for sq in sub_questions:
            if sq and sq not in variants:
                variants.append(sq)

        logger.info(f"🔄 Query Rewrite: {len(variants)} 个检索视角 -> {variants}")

        ranked_lists: List[List[Tuple[int, float]]] = []
        hit_bank: Dict[Any, Dict[str, Any]] = {}
        aggregate_info: List[Dict[str, Any]] = []

        for variant in variants:
            hits, variant_info = await self._retrieve_once(
                variant,
                top_k,
                tenant_uuid,
                allow_rerank=allow_rerank,
            )
            aggregate_info.append({
                "question": variant,
                "final_hits": len(hits),
                "mode": variant_info.get("fusion_mode", "bm25"),
                "reranked": variant_info.get("stage2") == "rerank",
            })
            ranked_lists.append([(self._get_hit_key(hit), 0.0) for hit in hits])
            for hit in hits:
                chunk_key = self._get_hit_key(hit)
                if chunk_key not in hit_bank:
                    hit_bank[chunk_key] = dict(hit)

        if not ranked_lists:
            return []

        fused_ranks = self._reciprocal_rank_fusion(ranked_lists)
        key = tenant_uuid or "default"
        texts = self._tenant_texts.get(key, [])

        result: List[Dict[str, Any]] = []
        for chunk_idx, vote_score in fused_ranks[:top_k]:
            hit = dict(hit_bank.get(chunk_idx, {}))
            hit["rewrite_score"] = vote_score
            result.append(hit)

        self._last_query_rewrite_info = aggregate_info  # debug / stage_info usage
        return result

    async def _generate_sub_questions(self, question: str, n: int) -> List[str]:
        """用 LLM 生成 N 个子问题（去重）"""
        prompt = f"""你是一个问题分解助手。请将下面的用户问题分解为 {n} 个不同角度的子问题。
每个子问题应该从不同角度探索原问题，以便全面检索相关信息。

用户问题：{question}

请输出 {n} 个子问题，每行一个，格式如下：
1. [子问题1]
2. [子问题2]
...

只输出子问题，不要其他解释。"""
        try:
            resp = await self.client_llm.chat.completions.create(
                model=self.settings.LLM_MODEL or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            raw = resp.choices[0].message.content
            lines = re.findall(r'\d+[\.、](.+?)$', raw, re.MULTILINE)
            sub_questions = [l.strip() for l in lines if l.strip()]
            return sub_questions[:n] if sub_questions else [question]
        except Exception as e:
            logger.warning(f"Query Rewrite 生成失败: {e}")
            return [question]

    # ==================== 主查询入口 ====================

    async def query(
        self,
        question: str,
        top_k: int = 5,
        tenant_uuid: Optional[int] = None,
        enable_rerank: bool = False,
        enable_query_rewrite: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        多阶段 RAG 查询

        Args:
            question: 查询问题
            top_k: 最终返回的 chunk 数量
            tenant_uuid: 租户ID
            enable_rerank: 是否启用 Stage2 Rerank
            enable_query_rewrite: 是否启用 Stage3 Query Rewrite

        Returns:
            (context_str, stage_info): 检索到的上下文 + 各阶段处理信息
        """
        import time
        self._ensure_tenant_ready(tenant_uuid)
        stage_info: Dict[str, Any] = {}
        t0 = time.time()
        t1 = time.time()
        stage_info["keyword_baseline"] = "bm25"
        stage_info["vector_available"] = self._has_vector_search()
        stage_info["rerank_available"] = self._has_rerank_model()

        if enable_query_rewrite and self.settings.ENABLE_QUERY_REWRITE and self._has_query_rewrite_model():
            allow_rerank = enable_rerank or self.settings.ENABLE_RERANK or self._has_rerank_model()
            final_hits = await self._stage3_query_rewrite(
                question,
                top_k,
                tenant_uuid,
                allow_rerank=allow_rerank,
            )
            stage_info["stage3"] = "query_rewrite_vote"
            stage_info["rewrite_enabled"] = True
            stage_info["rewrite_variants"] = getattr(self, "_last_query_rewrite_info", [])
        else:
            rerank_enabled = enable_rerank or self.settings.ENABLE_RERANK or self._has_rerank_model()
            final_hits, retrieval_info = await self._retrieve_once(
                question,
                top_k,
                tenant_uuid,
                allow_rerank=rerank_enabled,
            )
            stage_info.update(retrieval_info)
            stage_info["stage3"] = "none"
            stage_info["rewrite_enabled"] = False

        stage_info["stage1_ms"] = round((time.time() - t1) * 1000, 1)
        stage_info["total_ms"] = round((time.time() - t0) * 1000, 1)

        if not final_hits:
            return "未找到相关文档信息。", stage_info

        retrieved_texts = [hit["text"] for hit in final_hits]
        context_str = "\n\n---\n\n".join(retrieved_texts)

        return context_str, stage_info
