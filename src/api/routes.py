import logging
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel, Field
from src.utils.content_parse import parse_uploaded_file
from src.utils.setting_utils import save_config_to_file

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== 数据模型 (Pydantic) ====================
class ThinkingStep(BaseModel):
    """思考步骤"""
    step_type: str = Field(..., description="步骤类型: init/tool_start/tool_end/finish")
    title: str = Field(..., description="步骤标题")
    description: str = Field(..., description="步骤描述")
    timestamp: str = Field(..., description="时间戳")


class ToolCall(BaseModel):
    """工具调用记录"""
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None

class ConversationItem(BaseModel):
    """单条对话记录"""
    id: str
    timestamp: str
    user_content: str
    ai_content: str
    thinking_steps: List[ThinkingStep] = Field(default_factory=list, description="思考过程")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用记录")
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = 0
    duration_ms: Optional[int] = 0

class SavedSessionItem(BaseModel):
    """会话记录"""
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_count: int = 0
    conversation: List[ConversationItem] = Field(default_factory=list)  # 🔥 使用完整的 ConversationItem
    title: Optional[str] = "历史对话"
    statistics: Optional[Dict[str, Any]] = None  # 新增统计信息

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
    
@router.post("/chat/upload", response_model=ChatResponse, tags=["对话"])
async def chat_with_upload(
    request: Request,
    message: str = Form(...),                  # 接收 FormData 中的 message 字段
    session_id: Optional[str] = Form(None),    # 接收 FormData 中的 session_id 字段
    files: List[UploadFile] = File(...)        # 接收 FormData 中的 files 列表
):
    """
    支持附件上传的对话接口
    前端路径: /api/chat/upload
    Content-Type: multipart/form-data
    """
    engine = request.app.state.engine
    
    try:
        # 1. 获取会话
        session_id, agent = engine.get_or_create_session(session_id)
        
        logger.info(f"[{session_id}] 收到带附件的消息: {message[:50]}... (附件数: {len(files)})")

        # 2. 解析所有文件
        files_content = ""
        for file in files:
            file_content = await parse_uploaded_file(file)
            files_content += file_content
        
        # 3. 组合 Prompt
        # 将用户的问题和文件内容组合在一起
        full_prompt = f"{message}\n{files_content}"
        
        # 4. 调用 Agent (复用原有的 chat 逻辑)

        response_content = await agent.chat(full_prompt)

        final_answer = ""
        if hasattr(response_content, '__aiter__'):
            async for chunk in response_content:
                # 假设 chunk 是字符串，如果是对象需要根据实际情况取 .content
                if isinstance(chunk, str):
                    final_answer += chunk
                elif hasattr(chunk, 'content'): # 兼容某些框架的 chunk 对象
                    final_answer += chunk.content or ""
        else:
            # 如果是普通字符串，直接使用
            final_answer = str(response_content)
        
        return ChatResponse(
            message=final_answer,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"附件对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/history/saved", response_model=SavedHistoryListResponse, tags=["对话历史"])
async def get_saved_history_list(request: Request):
    """
    获取所有已保存的会话历史列表
    包含完整的思考过程和工具调用记录
    """
    engine = request.app.state.engine
    try:
        sessions = engine.history_manager.list_sessions()
        
        formatted_sessions = []
        for s in sessions:
            # 🔥 关键修改：完整解析每条对话，保留思考过程
            conversations = []
            for conv in s.get("conversation", []):
                # 解析思考步骤
                thinking_steps = []
                for step in conv.get("thinking_steps", []):
                    # 兼容旧格式（包含 timestamp 和 event_type）
                    if "event_type" in step and "data" in step:
                        # 旧格式：{"timestamp": "...", "event_type": "step", "data": {...}}
                        step_data = step["data"]
                        thinking_steps.append(ThinkingStep(
                            step_type=step_data.get("step", "unknown"),
                            title=step_data.get("title", "处理中"),
                            description=step_data.get("description", ""),
                            timestamp=step.get("timestamp", "")
                        ))
                    else:
                        # 新格式：直接包含字段
                        thinking_steps.append(ThinkingStep(
                            step_type=step.get("step_type", "unknown"),
                            title=step.get("title", "处理中"),
                            description=step.get("description", ""),
                            timestamp=step.get("timestamp", "")
                        ))
                
                # 解析工具调用
                tool_calls = [
                    ToolCall(**tc) for tc in conv.get("tool_calls", [])
                ]
                
                conversations.append(ConversationItem(
                    id=conv.get("id", ""),
                    timestamp=conv.get("timestamp", ""),
                    user_content=conv.get("user_content", ""),
                    ai_content=conv.get("ai_content", ""),
                    thinking_steps=thinking_steps,
                    tool_calls=tool_calls,
                    metadata=conv.get("metadata"),
                    tokens_used=conv.get("tokens_used", 0),
                    duration_ms=conv.get("duration_ms", 0)
                ))
            
            formatted_sessions.append(SavedSessionItem(
                session_id=s.get("session_id", "unknown"),
                created_at=s.get("created_at"),
                updated_at=s.get("updated_at"),
                conversation_count=s.get("conversation_count", 0),
                conversation=conversations,  # 🔥 使用完整解析的数据
                title=s.get("title", "历史对话"),
                statistics=s.get("statistics")
            ))

        return SavedHistoryListResponse(
            success=True,
            total=len(formatted_sessions),
            sessions=formatted_sessions
        )
    except Exception as e:
        logger.error(f"获取历史记录列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/session/{session_id}", tags=["对话历史"])
async def get_session_detail(session_id: str, request: Request):
    """
    获取单个会话的完整详情
    """
    engine = request.app.state.engine
    try:
        session_data = engine.history_manager.get_session_history(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "success": True,
            "session": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话详情失败: {e}", exc_info=True)
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
        engine.init_service()
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
    



