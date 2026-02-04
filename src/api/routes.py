"""
REST API 路由 - 优化版
优化要点：
1. 统一 rag_pipeline 管理，避免重复创建
2. settings 变化时自动重新初始化
3. 单例模式确保全局唯一实例

路径: src/api/routes.py
"""

import logging
import uuid
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import json
from config.settings import settings
from src.services.person_like_service import UserPreferenceMining

# GraphRAG 导入
try:
    from src.services.rag_graph import GraphRAGPipeline
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG 模块未找到，相关功能将不可用")

from src.api.websocket import ConversationHistoryManager

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 配置文件路径
CONFIG_FILE = Path("data/saved_config.json")
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = Path("data/history")


# ==================== 数据模型 ====================

class SavedSessionItem(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_count: int
    conversation: List[Dict[str, Any]]
    # ✅ 新增 title 字段，用于前端显示对话摘要
    title: Optional[str] = "新对话"

class SavedHistoryListResponse(BaseModel):
    success: bool
    total: int
    sessions: List[SavedSessionItem]

class SavedSessionDetailResponse(BaseModel):
    success: bool
    session_id: str
    conversations: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    session_id: str
    timestamp: str


class HistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    session_id: str



class StatusResponse(BaseModel):
    status: str
    message: str


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    document_name: str
    chunks_count: int
    uploaded_at: str


class DeleteResponse(BaseModel):
    success: bool
    message: str
    document_id: str


class QueryRequest(BaseModel):
    question: str
    top_k_communities: Optional[int] = 10


class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    processing_time: float


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
    total_entities: int
    total_relationships: int
    total_communities: int
    message: str


class ConfigResponse(BaseModel):
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None



class SavedSessionItem(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_count: int





# ==================== GraphRAG 管理器（优化版）====================

class GraphRAGManager:
    """
    GraphRAG 知识库管理器（单例模式）
    
    优化要点：
    1. 统一管理 rag_pipeline 实例
    2. 配置变化时自动重新初始化
    3. 延迟加载，只在需要时初始化
    """
    
    _instance = None
    _pipeline: Optional[GraphRAGPipeline] = None
    _config_hash: Optional[str] = None  # 用于检测配置变化
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化管理器"""
        if self._initialized:
            return
            
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        
        logger.info("📦 GraphRAGManager 初始化完成")
    
    def _get_config_hash(self) -> str:
        """获取当前配置的哈希值，用于检测配置变化"""
        config_str = f"{settings.LLM_API_KEY}|{settings.LLM_URL}|{settings.LLM_MODEL}|" \
                     f"{settings.EMBEDDING_API_KEY}|{settings.EMBEDDING_URL}|{settings.EMBEDDING_MODEL}|" \
                     f"{settings.EMBEDDING_DIM}|{settings.GRAPHRAG_STORAGE_DIR}"
        return str(hash(config_str))
    
    def _should_reinitialize(self) -> bool:
        """检查是否需要重新初始化（配置是否变化）"""
        current_hash = self._get_config_hash()
        if self._config_hash != current_hash:
            logger.info("🔄 检测到配置变化，需要重新初始化 GraphRAG")
            return True
        return False
    
    def _initialize_pipeline(self, force: bool = False):
        """
        初始化 GraphRAG Pipeline
        
        Args:
            force: 是否强制重新初始化
        """
        if not GRAPHRAG_AVAILABLE:
            logger.warning("⚠️ GraphRAG 模块不可用")
            return
        
        # 检查是否需要初始化
        if not force and self._pipeline is not None and not self._should_reinitialize():
            return
        
        try:
            logger.info("🔧 正在初始化 GraphRAG Pipeline...")
            
            # 创建新的 pipeline 实例
            self._pipeline = GraphRAGPipeline(
                llm_api_key=settings.LLM_API_KEY,
                embedding_api_key=settings.EMBEDDING_API_KEY,
                llm_url=settings.LLM_URL,
                embedding_url=settings.EMBEDDING_URL,
                embedding_name=settings.EMBEDDING_MODEL,
                embedding_dim=settings.EMBEDDING_DIM,
                llm_name=settings.LLM_MODEL,
                storage_dir=settings.GRAPHRAG_STORAGE_DIR
            )
            
            # 尝试加载已有知识库
            try:
                self._pipeline.load("default")
                logger.info("✅ GraphRAG: 加载已有知识库成功")
            except FileNotFoundError:
                logger.info("📝 GraphRAG: 创建新知识库")
            
            # 更新配置哈希
            self._config_hash = self._get_config_hash()
            
            logger.info("✅ GraphRAG Pipeline 初始化完成")
            
        except Exception as e:
            logger.error(f"❌ GraphRAG Pipeline 初始化失败: {e}", exc_info=True)
            self._pipeline = None
            raise
    
    def get_pipeline(self) -> GraphRAGPipeline:
        """
        获取 Pipeline 实例（懒加载 + 配置检测）
        
        Returns:
            GraphRAGPipeline 实例
            
        Raises:
            HTTPException: 如果 GraphRAG 不可用或初始化失败
        """
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="GraphRAG 模块不可用，请检查安装"
            )
        
        # 检查是否需要（重新）初始化
        if self._pipeline is None or self._should_reinitialize():
            self._initialize_pipeline()
        
        if self._pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="GraphRAG 服务未就绪，请检查配置"
            )
        
        return self._pipeline
    
    def force_reinitialize(self):
        """强制重新初始化（用于配置更新后）"""
        logger.info("🔄 强制重新初始化 GraphRAG...")
        self._pipeline = None
        self._config_hash = None
        self._initialize_pipeline(force=True)
    
    def is_ready(self) -> bool:
        """检查是否就绪"""
        try:
            pipeline = self.get_pipeline()
            return pipeline is not None
        except:
            return False
    
    def save_upload_file(self, upload_file: UploadFile) -> Path:
        """保存上传文件"""
        file_path = self.upload_dir / f"{uuid.uuid4()}_{upload_file.filename}"
        
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            return file_path
        finally:
            upload_file.file.close()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.is_ready():
            return {
                'enabled': False,
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'total_communities': 0,
                'index_status': 'Not initialized'
            }
        
        try:
            pipeline = self.get_pipeline()
            total_communities = sum(len(comms) for comms in pipeline.communities.values())
            is_indexed = pipeline.community_summary_index is not None
            
            return {
                'enabled': True,
                'total_documents': len(pipeline.documents),
                'total_entities': len(pipeline.entities),
                'total_relationships': len(pipeline.relationships),
                'total_communities': total_communities,
                'index_status': 'Indexed' if is_indexed else 'Not indexed'
            }
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {
                'enabled': False,
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'total_communities': 0,
                'index_status': f'Error: {str(e)}'
            }
    
    def save(self):
        """保存知识库"""
        try:
            pipeline = self.get_pipeline()
            pipeline.save("default")
            logger.info("✅ GraphRAG 知识库已保存")
        except Exception as e:
            logger.error(f"❌ 保存知识库失败: {e}")
            raise


# ==================== Session 管理器 ====================

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_timestamps = {}
    def get_or_create_session(self, session_id=None):
        if session_id and session_id in self.sessions: return session_id, self.sessions[session_id]
        sid = str(uuid.uuid4())
        self.sessions[sid] = ConversationalAgent()
        return sid, self.sessions[sid]
    def clear_session(self, sid):
        if sid in self.sessions: self.sessions[sid].clear_history()
    def delete_session(self, sid):
        if sid in self.sessions: del self.sessions[sid]
    def cleanup_old_sessions(self, timeout): pass
    def get_session_count(self): return len(self.sessions)

def load_config_from_file(): return {}
def save_config_to_file(cfg): return {}


# ==================== 全局管理器实例 ====================

session_manager = SessionManager()
graphrag_manager = GraphRAGManager()


# ==================== 辅助函数 ====================

def load_history_from_file() -> List[Dict[str, Any]]:
    """从文件加载历史记录"""
    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.error(f"加载历史记录失败: {e}")
            return []
    return []


def load_config_from_file() -> Dict[str, Any]:
    """从 JSON 文件加载配置"""
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
    
    return {
        "llm_config": {},
        "search_config": {},
        "firecrawl_config": {},
        "embedding_config": {}
    }


def save_config_to_file(new_flat_config: Dict[str, str]) -> Dict[str, Any]:
    """保存配置到文件"""
    current_config = load_config_from_file()
    
    field_mapping = {
        'LLM_API_KEY': 'llm_config',
        'LLM_URL': 'llm_config',
        'LLM_MODEL': 'llm_config',
        'TAVILY_API_KEY': 'search_config',
        'FIRECRAWL_API_KEY': 'firecrawl_config',
        'FIRECRAWL_URL': 'firecrawl_config',
        'EMBEDDING_API_KEY': 'embedding_config',
        'EMBEDDING_URL': 'embedding_config',
        'EMBEDDING_MODEL': 'embedding_config'
    }
    
    for key, value in new_flat_config.items():
        if not value or ("..." in value and len(value) < 20):
            continue
        
        group = field_mapping.get(key)
        if group:
            if group not in current_config:
                current_config[group] = {}
            current_config[group][key] = value
    
    try:
        with CONFIG_FILE.open('w', encoding='utf-8') as f:
            json.dump(current_config, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ 配置已保存到文件: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"❌ 保存配置文件失败: {e}")
        raise
    
    return current_config


async def rebuild_index_background():
    """后台任务：重建索引"""
    try:
        logger.info("📊 开始重建 GraphRAG 索引...")
        pipeline = graphrag_manager.get_pipeline()
        await pipeline.rebuild_index()
        graphrag_manager.save()
        logger.info("✅ GraphRAG 索引重建完成")
    except Exception as e:
        logger.error(f"❌ 重建索引失败: {e}", exc_info=True)



@router.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """同步对话接口"""
    try:
        session_manager.cleanup_old_sessions(settings.SESSION_TIMEOUT)
        session_id, agent = session_manager.get_or_create_session(request.session_id)
        
        logger.info(f"[{session_id}] 收到消息: {request.message[:50]}...")
        
        response_content = await agent.chat(request.message)
        
        return ChatResponse(
            message=response_content,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/saved", response_model=SavedHistoryListResponse, tags=["对话历史"])
async def get_saved_history_list():
    """获取所有已保存的会话历史列表"""
    try:
        # 调用刚刚修改过的 list_sessions

        history_manager=ConversationHistoryManager()
        sessions = history_manager.list_sessions()
        
        return SavedHistoryListResponse(
            success=True,
            total=len(sessions),
            sessions=sessions
        )
    except Exception as e:
        logger.error(f"获取历史记录列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["对话"])
async def get_history(session_id: str):
    """获取对话历史"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = session_manager.sessions[session_id]
    
    return HistoryResponse(
        history=agent.get_history(),
        session_id=session_id
    )

@router.post("/clear/{session_id}", response_model=StatusResponse, tags=["对话"])
async def clear_history(session_id: str):
    """清空对话历史"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.clear_session(session_id)
    
    return StatusResponse(
        status="ok",
        message="History cleared"
    )


@router.delete("/session/{session_id}", response_model=StatusResponse, tags=["对话"])
async def delete_session(session_id: str):
    """删除 Session"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.delete_session(session_id)
    
    return StatusResponse(
        status="ok",
        message="Session deleted"
    )


# ==================== GraphRAG 接口 ====================

@router.post("/graphrag/documents/upload", response_model=UploadResponse, tags=["GraphRAG"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    auto_rebuild: bool = True
):
    """上传文档到知识库"""
    
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_extension}"
        )
    
    try:
        # 保存文件
        file_path = graphrag_manager.save_upload_file(file)
        logger.info(f"📄 文件已保存: {file_path}")
        
        # 准备元数据
        metadata = {
            "title": title or file.filename,
            "author": author,
            "category": category,
            "original_filename": file.filename,
            "file_extension": file_extension,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # ★ 使用统一的 pipeline
        pipeline = graphrag_manager.get_pipeline()
        doc_uuid = await pipeline.add_document(str(file_path), metadata)
        doc_info = pipeline.documents[pipeline.uuid_to_docid[doc_uuid]]
        
        # 后台重建索引
        if auto_rebuild:
            background_tasks.add_task(rebuild_index_background)
        
        return UploadResponse(
            success=True,
            message="文档上传成功" + ("，正在后台重建索引" if auto_rebuild else ""),
            document_id=doc_uuid,
            document_name=file.filename,
            chunks_count=len(doc_info['chunks']),
            uploaded_at=doc_info['added_at']
        )
        
    except Exception as e:
        logger.error(f"❌ 上传文档失败: {e}", exc_info=True)
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.get("/graphrag/documents", response_model=ListDocumentsResponse, tags=["GraphRAG"])
async def list_documents():
    """获取所有文档列表"""
    try:
        # ★ 使用统一的 pipeline
        pipeline = graphrag_manager.get_pipeline()
        docs =  pipeline.list_documents()
        
        documents = [
            DocumentInfo(
                document_id=doc['uuid'],
                name=doc['name'],
                path=doc['path'],
                chunks=doc['chunks'],
                uploaded_at=doc['added_at'],
                metadata=doc['metadata']
            )
            for doc in docs
        ]
        
        return ListDocumentsResponse(
            success=True,
            total=len(documents),
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"❌ 获取文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/graphrag/index/rebuild", tags=["GraphRAG"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """手动触发索引重建"""
    stats = graphrag_manager.get_stats()
    if stats['total_documents'] == 0:
        raise HTTPException(status_code=400, detail="没有文档可以索引")
    
    try:
        background_tasks.add_task(rebuild_index_background)
        
        return {
            "success": True,
            "message": "索引重建任务已添加到后台队列",
            "total_documents": stats['total_documents']
        }
        
    except Exception as e:
        logger.error(f"❌ 触发索引重建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"触发索引重建失败: {str(e)}")


@router.get("/graphrag/index/status", response_model=IndexStatusResponse, tags=["GraphRAG"])
async def get_index_status():
    """获取索引状态"""
    try:
        stats = graphrag_manager.get_stats()
        is_indexed = stats['index_status'] == 'Indexed'
        
        return IndexStatusResponse(
            success=True,
            is_indexed=is_indexed,
            total_documents=stats['total_documents'],
            total_entities=stats['total_entities'],
            total_relationships=stats['total_relationships'],
            total_communities=stats['total_communities'],
            message=stats['index_status']
        )
        
    except Exception as e:
        logger.error(f"❌ 获取索引状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取索引状态失败: {str(e)}")


@router.post("/graphrag/query", response_model=QueryResponse, tags=["GraphRAG"])
async def query_knowledge_base(request: QueryRequest):
    """查询知识库"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        start_time = datetime.now()
        
        # ★ 使用统一的 pipeline
        pipeline = graphrag_manager.get_pipeline()
        answer = await pipeline.global_query(
            request.question,
            top_k_communities=request.top_k_communities
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ 查询完成: {request.question[:50]}... (耗时 {processing_time:.2f}s)")
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=answer,
            processing_time=processing_time
        )
        
    except RuntimeError as e:
        if "索引未构建" in str(e):
            raise HTTPException(
                status_code=400,
                detail="索引未构建，请先上传文档并等待索引构建完成"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ==================== 配置管理接口 ====================

@router.get("/config", response_model=ConfigResponse, tags=["配置管理"])
async def get_config():
    """获取当前配置"""
    try:
        config_data = load_config_from_file()
        
        # 深拷贝并脱敏
        safe_config = json.loads(json.dumps(config_data))
        
        for group, items in safe_config.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    if value and isinstance(value, str) and 'KEY' in key.upper() and len(value) > 8:
                        items[key] = f"{value[:4]}...{value[-4:]}"
        
        return ConfigResponse(
            success=True,
            message="配置获取成功",
            config=safe_config
        )
        
    except Exception as e:
        logger.error(f"❌ 获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.post("/config", response_model=ConfigResponse, tags=["配置管理"])
async def update_config(config: Dict[str, str]):
    """更新配置"""
    try:
        # 保存配置
        updated_nested_config = save_config_to_file(config)
        
        # 更新 settings
        flat_settings = {}
        for group in updated_nested_config.values():
            if isinstance(group, dict):
                flat_settings.update(group)
        settings.update(**flat_settings)
        
        # ★★★ 关键：强制重新初始化 GraphRAG ★★★
        try:
            graphrag_manager.force_reinitialize()
            logger.info("✅ GraphRAG 已使用新配置重新初始化")
        except Exception as e:
            logger.warning(f"⚠️ GraphRAG 重新初始化失败（可能配置不完整）: {e}")
        
        logger.info(f"✅ 配置已更新并应用")
        
        return ConfigResponse(
            success=True,
            message="配置保存成功",
            config=None
        )
        
    except Exception as e:
        logger.error(f"❌ 更新配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


# ==================== 服务信息接口 ====================

@router.get("/info", tags=["服务信息"])
async def get_info():
    """获取服务信息"""
    graphrag_stats = graphrag_manager.get_stats()
    
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "features": {
            "streaming": True,
            "graphrag": graphrag_stats['enabled'],
            "config_management": True
        },
        "endpoints": {
            "http": {
                "chat": "/api/chat",
                "config_get": "/api/config",
                "config_update": "/api/config"
            }
        },
        "graphrag_status": graphrag_stats
    }


@router.get("/stats", tags=["服务信息"])
async def get_stats():
    """获取服务统计信息"""
    graphrag_stats = graphrag_manager.get_stats()
    
    return {
        "active_sessions": session_manager.get_session_count(),
        "graphrag": graphrag_stats,
        "config_loaded": CONFIG_FILE.exists(),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/person_like", tags=["个人偏好挖掘"])
async def get_person_like():
    mining_enginer=UserPreferenceMining()
    return mining_enginer.get_frontend_format()
