import logging
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel
from src.utils.setting_utils import save_config_to_file

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== 数据模型 (Pydantic) ====================

class SavedSessionItem(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_count: int
    conversation: List[Dict[str, Any]] = []
    title: Optional[str] = "新对话"

class SavedHistoryListResponse(BaseModel):
    success: bool
    total: int
    sessions: List[SavedSessionItem]

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
    top_k: int = 5

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

class ConfigResponse(BaseModel):
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None


class IndexStatusResponse(BaseModel):
    success: bool
    document_count: int
    status: str = "ready"

# ==================== API 路由实现 ====================

@router.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: Request, chat_req: ChatRequest):
    """同步对话接口"""
    engine = request.app.state.engine  # 从全局状态获取唯一引擎
    try:
        session_id, agent = engine.get_or_create_session(chat_req.session_id)
        
        logger.info(f"[{session_id}] 收到消息: {chat_req.message[:50]}...")
        response_content = await agent.chat(chat_req.message)
        
        return ChatResponse(
            message=response_content,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/saved", response_model=SavedHistoryListResponse, tags=["对话历史"])
async def get_saved_history_list(request: Request):
    """获取所有已保存的会话历史列表"""
    engine = request.app.state.engine
    try:
        sessions = engine.history_manager.list_sessions()
        
        formatted_sessions = [
            SavedSessionItem(
                session_id=s.get("session_id", "unknown"),
                created_at=s.get("created_at"),
                updated_at=s.get("updated_at"),
                conversation_count=s.get("conversation_count", 0),
                conversation=s.get("conversation", []),
                title=s.get("title", "历史对话")
            ) for s in sessions
        ]

        return SavedHistoryListResponse(
            success=True,
            total=len(formatted_sessions),
            sessions=formatted_sessions
        )
    except Exception as e:
        logger.error(f"获取历史记录列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["对话历史"])
async def get_history(request: Request, session_id: str):
    """获取内存中的对话历史"""
    engine = request.app.state.engine
    if session_id not in engine.sessions:
        raise HTTPException(status_code=404, detail="Session active not found")
    
    agent = engine.sessions[session_id]
    return HistoryResponse(
        history=agent.get_history(),
        session_id=session_id
    )

@router.delete("/session/{session_id}", response_model=StatusResponse, tags=["对话"])
async def delete_session(request: Request, session_id: str):
    """删除内存中的 Session"""
    engine = request.app.state.engine
    if session_id in engine.sessions:
        del engine.sessions[session_id]
        return StatusResponse(status="ok", message="Session deleted from memory")
    raise HTTPException(status_code=404, detail="Session not found")

@router.post("/rag/documents/upload", response_model=UploadResponse, tags=["RAG"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None
):
    """上传文档"""
    engine = request.app.state.engine
    allowed = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
    
    try:
        file_path = engine.settings.UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        metadata = {
            "title": title or file.filename,
            "author": author,
            "category": category,
            "original_filename": file.filename,
            "file_extension": ext
        }
        
        pipeline = engine.rag_engine
        doc_uuid = await pipeline.add_document(str(file_path), metadata)
        doc_info = pipeline.documents.get(doc_uuid, {'chunks_count': 0, 'added_at': str(datetime.now())})
        
        return UploadResponse(
            success=True,
            message="文档上传并索引成功",
            document_id=doc_uuid,
            document_name=file.filename,
            chunks_count=doc_info.get('chunks_count', 0),
            uploaded_at=doc_info.get('added_at', "")
        )
    except Exception as e:
        logger.error(f"❌ 上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/documents", response_model=ListDocumentsResponse, tags=["RAG"])
async def list_documents(request: Request):
    """获取文档列表"""
    engine = request.app.state.engine
    try:
        docs = engine.rag_engine.list_documents()
        doc_list = [
            DocumentInfo(
                document_id=doc['uuid'],
                name=doc['title'],
                path=str(doc.get('path', '')),
                chunks=doc.get('chunks_count', 0),
                uploaded_at=doc.get('added_at', ''),
                metadata=doc
            ) for doc in docs
        ]
        return ListDocumentsResponse(success=True, total=len(doc_list), documents=doc_list)
    except Exception as e:
        logger.error(f"列表获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rag/documents/{document_id}", response_model=DeleteResponse, tags=["RAG"])
async def delete_document(request: Request, document_id: str):
    """删除文档"""
    engine = request.app.state.engine
    try:
        engine.rag_engine.remove_document(document_id)
        return DeleteResponse(success=True, message="文档已删除", document_id=document_id)
    except Exception as e:
        logger.error(f"❌ 删除失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/query", response_model=QueryResponse, tags=["RAG"])
async def query_knowledge_base(request: Request, query_req: QueryRequest):
    """查询知识库"""
    engine = request.app.state.engine
    if not query_req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        start_time = datetime.now()
        answer = await engine.rag_engine.query(query_req.question, top_k=query_req.top_k)
        duration = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=True,
            question=query_req.question,
            answer=str(answer),
            processing_time=duration
        )
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/rag/index/status", response_model=IndexStatusResponse, tags=["RAG"])
async def get_index_status(request: Request):
    """获取 RAG 索引状态（当前仅返回文档总数）"""
    engine = request.app.state.engine
    try:
        docs = engine.rag_engine.list_documents()

        return IndexStatusResponse(
            success=True,
            document_count=len(docs),
            status="ready"
        )
    except Exception as e:
        logger.error(f"获取索引状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/config", response_model=ConfigResponse, tags=["配置管理"])
async def get_config(request: Request):
    """读取配置"""
    engine = request.app.state.engine
    try:
        if engine.settings.CONFIG_FILE.exists():
            data = json.loads(engine.settings.CONFIG_FILE.read_text(encoding='utf-8'))
            return ConfigResponse(success=True, message="OK", config=data)
        return ConfigResponse(success=True, message="Empty Config", config={})
    except Exception as e:
        return ConfigResponse(success=False, message=str(e))

@router.post("/config", response_model=ConfigResponse, tags=["配置管理"])
async def update_config(request: Request, config: Dict[str, Any]):
    """更新配置"""
    engine = request.app.state.engine
    try:
        updated_nested_config = save_config_to_file(config)
        
        return ConfigResponse(success=True, message="配置已保存，在新的对话中生效。", config=updated_nested_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/person_like", tags=["个人偏好挖掘"])
async def get_person_like(request: Request):
    engine = request.app.state.engine
    try:
        return engine.user_preference.get_frontend_format()
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
    



