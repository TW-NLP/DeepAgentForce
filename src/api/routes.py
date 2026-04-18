import logging
import uuid
import shutil
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, BackgroundTasks, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.utils.content_parse import parse_uploaded_file
from src.utils.setting_utils import save_config_to_file
from src.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 多租户辅助函数 ====================

def get_tenant_uuid_from_request(request: Request) -> Optional[str]:
    """
    从请求头中提取 tenant_uuid
    用于 API 路由的租户隔离验证
    优先级：X-Tenant-UUID header > Authorization JWT token
    """
    # 🆕 优先从 X-Tenant-UUID header 获取（供 skill 脚本使用）
    tenant_header = request.headers.get("X-Tenant-UUID")
    if tenant_header:
        return tenant_header

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.replace("Bearer ", "")
    try:
        payload = auth_service.verify_token(token)
        return payload.get("tenant_uuid")  # 🆕 使用 tenant_uuid
    except Exception:
        return None




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
    chunks: int = 0
    size: int = 0
    uploaded_at: str = ""
    metadata: Optional[Dict] = {}

class ListDocumentsResponse(BaseModel):
    success: bool
    total: int
    documents: List[DocumentInfo]

class OutputFileInfo(BaseModel):
    """输出文件信息"""
    name: str
    path: str
    size: int
    modified_at: str
    is_directory: bool = False

class ListOutputFilesResponse(BaseModel):
    success: bool
    files: List[OutputFileInfo]
    current_path: str


class ConfigResponse(BaseModel):
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None


class IndexStatusResponse(BaseModel):
    success: bool
    document_count: int
    chunks_count: int = 0
    total_size: str = "0 KB"
    last_updated: Optional[str] = None
    status: str = "ready"

# ==================== API 路由实现 ====================

@router.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: Request, chat_req: ChatRequest):
    """同步对话接口"""
    engine = request.app.state.engine  # 从全局状态获取唯一引擎
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        session_id, agent = engine.get_or_create_session(
            chat_req.session_id,
            tenant_uuid=tenant_id,  # 🆕
        )

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
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕

    try:
        # 1. 获取会话
        session_id, agent = engine.get_or_create_session(
            session_id,
            tenant_uuid=tenant_id,  # 🆕
        )
        
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


async def generate_upload_chat_stream(agent, session_id, full_prompt, message, files, tenant_id=None):
    """生成上传文件对话的流式响应"""
    try:
        # 发送开始消息
        yield f"event: start\ndata: {json.dumps({'session_id': session_id})}\n\n"

        # 获取 agent 的流式响应
        response_content = await agent.chat(full_prompt)

        full_answer = ""
        step_count = 0

        if hasattr(response_content, '__aiter__'):
            async for chunk in response_content:
                if isinstance(chunk, str):
                    full_answer += chunk
                elif hasattr(chunk, 'content'):
                    full_answer += chunk.content or ""
                elif isinstance(chunk, dict):
                    # 处理事件类型的数据
                    chunk_type = chunk.get('type', '')
                    if chunk_type == 'step':
                        step_count += 1
                        step_data = chunk.get('data', {})
                        yield f"event: step\ndata: {json.dumps({'step': step_count, 'title': step_data.get('title', ''), 'description': step_data.get('description', '')})}\n\n"
                    elif chunk_type == 'tool':
                        yield f"event: tool\ndata: {json.dumps(chunk.get('data', {}))}\n\n"
                    else:
                        full_answer += str(chunk)
                else:
                    full_answer += str(chunk)

                # 定期发送部分答案
                if len(full_answer) > 0 and len(full_answer) % 50 == 0:
                    yield f"event: chunk\ndata: {json.dumps({'content': full_answer})}\n\n"

        # 发送最终答案
        yield f"event: done\ndata: {json.dumps({'message': full_answer, 'session_id': session_id})}\n\n"

    except Exception as e:
        logger.error(f"流式对话失败: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@router.post("/chat/upload/stream")
async def chat_with_upload_stream(
    request: Request,
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    支持附件上传的对话接口（流式版本）
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)

    try:
        # 1. 获取会话
        session_id, agent = engine.get_or_create_session(
            session_id,
            tenant_uuid=tenant_id,
        )

        logger.info(f"[{session_id}] 收到带附件的流式消息: {message[:50]}... (附件数: {len(files)})")

        # 2. 解析所有文件
        files_content = ""
        for file in files:
            file_content = await parse_uploaded_file(file)
            files_content += file_content

        # 3. 组合 Prompt
        full_prompt = f"{message}\n{files_content}"

        # 4. 返回流式响应
        return StreamingResponse(
            generate_upload_chat_stream(agent, session_id, full_prompt, message, files, tenant_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"流式附件对话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/sessions", tags=["对话历史"])
async def get_sessions_list(request: Request):
    """
    获取会话列表（轻量版，用于侧边栏显示）
    🆕 多租户：只返回当前用户所属租户的会话
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)
    try:
        sessions = engine.history_manager.list_sessions(tenant_uuid=tenant_id)

        # 返回轻量版会话信息
        session_list = []
        for s in sessions:
            # 获取第一条用户消息作为标题
            first_message = ""
            conversation = s.get("conversation", [])
            if conversation and len(conversation) > 0:
                for conv in conversation:
                    user_content = conv.get("user_content", "")
                    if user_content:
                        first_message = user_content[:50]
                        break

            session_list.append({
                "session_id": s.get("session_id", "unknown"),
                "title": s.get("title") or first_message or "新对话",
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "conversation_count": s.get("conversation_count", 0)
            })

        # 按更新时间倒序
        session_list.sort(key=lambda x: x.get("updated_at") or "", reverse=True)

        return {"sessions": session_list}
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/sessions/{session_id}/messages", tags=["对话历史"])
async def get_session_messages(session_id: str, request: Request):
    """
    获取指定会话的消息列表
    🆕 多租户：验证会话属于当前租户
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)
    try:
        session_data = engine.history_manager.get_session_history(session_id, tenant_uuid=tenant_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="会话不存在或无权限访问")

        # 提取消息列表
        messages = []
        for conv in session_data.get("conversations", []):
            user_content = conv.get("user_content", "")
            ai_content = conv.get("ai_content", "")
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })
            if ai_content:
                messages.append({
                    "role": "assistant",
                    "content": ai_content
                })

        return {"messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/saved", response_model=SavedHistoryListResponse, tags=["对话历史"])
async def get_saved_history_list(request: Request):
    """
    获取当前租户下所有已保存的会话历史列表
    包含完整的思考过程和工具调用记录
    🆕 多租户：只返回当前用户所属租户的会话
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        sessions = engine.history_manager.list_sessions(tenant_uuid=tenant_id)  # 🆕
        
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
    🆕 多租户：验证会话属于当前租户
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        session_data = engine.history_manager.get_session_history(session_id, tenant_uuid=tenant_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="会话不存在或无权限访问")
        
        return {
            "success": True,
            "session": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/session/{session_id}", tags=["对话历史"])
async def delete_session(session_id: str, request: Request):
    """
    删除指定会话
    🆕 多租户：验证会话属于当前租户
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        # 🆕 从内存中移除（如果存在，按租户 key）
        session_key = f"{tenant_id}_{session_id}" if tenant_id else session_id
        if session_key in engine.sessions:
            del engine.sessions[session_key]

        # 从磁盘删除（按租户目录）
        success = engine.history_manager.delete_session(session_id, tenant_uuid=tenant_id)

        if success:
            return {"success": True, "message": "会话已删除"}
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["对话历史"])
async def get_history(request: Request, session_id: str):
    """获取内存中的对话历史"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    session_key = f"{tenant_id}_{session_id}" if tenant_id else session_id
    if session_key not in engine.sessions:
        raise HTTPException(status_code=404, detail="Session active not found")
    
    agent = engine.sessions[session_key]
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
    """
    上传文档到当前租户的知识库
    🆕 多租户：文档按租户隔离
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    allowed = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
    
    try:
        # 🆕 多租户：文件存储到租户专属目录
        tenant_upload_dir = engine.settings.UPLOAD_DIR / str(tenant_id or "default")
        tenant_upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = tenant_upload_dir / f"{uuid.uuid4()}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        metadata = {
            "title": title or file.filename,
            "author": author,
            "category": category,
            "original_filename": file.filename,
            "file_extension": ext,
            "tenant_uuid": tenant_id,  # 🆕 记录所属租户
        }
        
        # 🆕 获取租户专属的 RAG pipeline
        pipeline = engine.get_rag_engine(tenant_id)
        doc_uuid = await pipeline.add_document(str(file_path), metadata, tenant_uuid=tenant_id)
        # 🆕 从租户文档元数据获取信息
        doc_info = pipeline._get_tenant_documents(tenant_id).get(doc_uuid, {'chunks_count': 0, 'added_at': str(datetime.now())})
        
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
    """获取当前租户的知识库文档列表"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        pipeline = engine.get_rag_engine(tenant_id)
        docs = pipeline.list_documents(tenant_uuid=tenant_id)  # 🆕
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
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        pipeline = engine.get_rag_engine(tenant_id)
        pipeline.remove_document(document_id, tenant_uuid=tenant_id)  # 🆕
        return DeleteResponse(success=True, message="文档已删除", document_id=document_id)
    except Exception as e:
        logger.error(f"❌ 删除失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/documents/{document_id}/content", tags=["RAG"])
async def get_document_content(request: Request, document_id: str):
    """获取文档内容预览"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        pipeline = engine.get_rag_engine(tenant_id)
        docs = pipeline.list_documents(tenant_uuid=tenant_id)

        # 找到指定文档 (uuid 是存储的字段名)
        doc = None
        doc_name = None
        uploaded_at = None
        for d in docs:
            if d.get('uuid') == document_id:
                doc = d
                doc_name = d.get('title', '')
                uploaded_at = d.get('added_at', '')
                break

        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 从 Milvus 获取文档的所有 chunks
        try:
            collection_name = pipeline._get_tenant_collection(tenant_id)
            milvus_results = pipeline.milvus.query(
                collection_name=collection_name,
                filter=f'doc_id == "{document_id}"',
                output_fields=["text"]
            )
            content_chunks = [{'content': item.get('text', '')} for item in milvus_results if item.get('text')]
        except Exception as e:
            logger.warning(f"从 Milvus 获取 chunks 失败: {e}")
            content_chunks = []

        return {
            "success": True,
            "document_id": document_id,
            "name": doc_name,
            "chunks": content_chunks,
            "uploaded_at": uploaded_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档内容失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/query", response_model=QueryResponse, tags=["RAG"])
async def query_knowledge_base(request: Request, query_req: QueryRequest):
    """查询知识库"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    if not query_req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        start_time = datetime.now()
        pipeline = engine.get_rag_engine(tenant_id)
        answer = await pipeline.query(
            query_req.question,
            top_k=query_req.top_k,
            tenant_uuid=tenant_id  # 🆕
        )
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
    """获取 RAG 索引状态"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        pipeline = engine.get_rag_engine(tenant_id)
        docs = pipeline.list_documents(tenant_uuid=tenant_id)  # 🆕

        # 计算总块数和大小
        total_chunks = sum(doc.get('chunks', 0) for doc in docs)
        total_size = sum(doc.get('size', 0) for doc in docs)

        # 格式化大小
        def format_size(bytes_val):
            if bytes_val == 0:
                return "0 B"
            k = 1024
            sizes = ['B', 'KB', 'MB', 'GB']
            i = int(math.floor(math.log(bytes_val) / math.log(k)))
            return f"{bytes_val / k ** i:.1f} {sizes[i]}"

        # 获取最后更新时间
        last_updated = None
        if docs:
            uploaded_times = [doc.get('uploaded_at', '') for doc in docs if doc.get('uploaded_at')]
            if uploaded_times:
                last_updated = max(uploaded_times)

        return IndexStatusResponse(
            success=True,
            document_count=len(docs),
            chunks_count=total_chunks,
            total_size=format_size(total_size),
            last_updated=last_updated,
            status="ready"
        )
    except Exception as e:
        logger.error(f"获取索引状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/config", response_model=ConfigResponse, tags=["配置管理"])
async def get_config(request: Request):
    """
    读取当前租户的配置
    🆕 多租户：每个租户有独立的配置
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        # 🆕 按租户获取配置文件路径
        config_file = engine.settings.get_tenant_config_file(tenant_id)
        if config_file.exists():
            data = json.loads(config_file.read_text(encoding='utf-8'))
            return ConfigResponse(success=True, message="OK", config=data)
        return ConfigResponse(success=True, message="Empty Config", config={})
    except Exception as e:
        return ConfigResponse(success=False, message=str(e))

@router.post("/config", response_model=ConfigResponse, tags=["配置管理"])
async def update_config(request: Request, config: Dict[str, Any]):
    """
    更新当前租户的配置
    🆕 多租户：每个租户有独立的配置
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        logger.info(f"[Config] 收到配置更新请求，tenant={tenant_id}, config={config}")
        updated_nested_config = save_config_to_file(config, tenant_uuid=tenant_id)  # 🆕
        logger.info(f"[Config] 配置已保存: {updated_nested_config}")
        engine.init_service(tenant_uuid=tenant_id)  # 🆕
        return ConfigResponse(success=True, message="配置已保存，在新的对话中生效。", config=updated_nested_config)
    except Exception as e:
        logger.error(f"[Config] 保存配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/person_like", tags=["个人偏好挖掘"])
async def get_person_like(request: Request):
    """获取当前用户的画像"""
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        return engine.user_preference.get_frontend_format(tenant_uuid=tenant_id)  # 🆕
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@router.get("/output/files", response_model=ListOutputFilesResponse, tags=["Output 文件管理"])
async def list_output_files(request: Request, path: Optional[str] = None):
    """
    获取 output 目录下的文件列表
    🆕 多租户：按租户隔离
    path: 可选的子目录路径（相对于 OUTPUT_DIR/{tenant_id}/）
    """
    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        # 🆕 多租户：基础路径包含租户目录
        base_path = engine.settings.get_tenant_output_dir(tenant_id)
        if path:
            current_path = base_path / path
            # 安全检查：确保路径在租户输出目录内
            if not str(current_path.resolve()).startswith(str(base_path.resolve())):
                raise HTTPException(status_code=400, detail="非法路径")
        else:
            current_path = base_path

        if not current_path.exists():
            current_path.mkdir(parents=True, exist_ok=True)
            return ListOutputFilesResponse(success=True, files=[], current_path=str(current_path))

        files = []
        for item in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            files.append(OutputFileInfo(
                name=item.name,
                path=str(item.relative_to(base_path)),
                size=item.stat().st_size if item.is_file() else 0,
                modified_at=datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                is_directory=item.is_dir()
            ))

        return ListOutputFilesResponse(
            success=True,
            files=files,
            current_path=str(current_path.relative_to(base_path)) or "."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 output 文件列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/output/files/download", tags=["Output 文件管理"])
async def download_output_file(request: Request, path: str):
    """
    下载 output 目录下的指定文件
    🆕 多租户：按租户隔离
    path: 文件路径（相对于 OUTPUT_DIR/{tenant_id}/）
    """
    from fastapi.responses import FileResponse

    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        # 🆕 多租户：基础路径包含租户目录
        base_path = engine.settings.get_tenant_output_dir(tenant_id)
        file_path = base_path / path
        # 安全检查：确保路径在租户输出目录内
        if not str(file_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=400, detail="非法路径")

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/output/files/preview", tags=["Output 文件管理"])
async def preview_output_file(request: Request, path: str):
    """
    预览 output 目录下的文本文件内容
    🆕 多租户：按租户隔离
    path: 文件路径（相对于 OUTPUT_DIR/{tenant_id}/）
    """
    from fastapi.responses import PlainTextResponse

    engine = request.app.state.engine
    tenant_id = get_tenant_uuid_from_request(request)  # 🆕
    try:
        # 🆕 多租户：基础路径包含租户目录
        base_path = engine.settings.get_tenant_output_dir(tenant_id)
        file_path = base_path / path
        # 安全检查：确保路径在租户输出目录内
        if not str(file_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=400, detail="非法路径")

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")

        # 检查是否为文本文件（简单的白名单检查）
        text_extensions = {
            '.txt', '.md', '.markdown', '.py', '.js', '.ts', '.jsx', '.tsx',
            '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.scss',
            '.csv', '.log', '.sh', '.bat', '.ini', '.conf', '.cfg', '.toml',
            '.sql', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs',
            '.vue', '.svelte', '.rb', '.php', '.swift', '.kt', '.scala',
        }
        ext = file_path.suffix.lower()
        if ext not in text_extensions and not _is_likely_text(file_path):
            raise HTTPException(status_code=400, detail="不支持预览的文件类型")

        # 读取文件内容（限制大小）
        max_size = 2 * 1024 * 1024  # 2MB
        if file_path.stat().st_size > max_size:
            content = f"文件过大 ({file_path.stat().st_size // 1024}KB)，无法预览。\n请下载后查看。"
        else:
            try:
                content = file_path.read_text(encoding='utf-8')
                # 限制返回行数
                lines = content.split('\n')
                if len(lines) > 500:
                    content = '\n'.join(lines[:500]) + f"\n\n... (共 {len(lines)} 行，内容过长已截断)"
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='gbk')
                    if len(content.split('\n')) > 500:
                        lines = content.split('\n')
                        content = '\n'.join(lines[:500]) + f"\n\n... (共 {len(lines)} 行，内容过长已截断)"
                except Exception:
                    raise HTTPException(status_code=400, detail="无法读取文件内容，可能是二进制文件")

        return PlainTextResponse(content=content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预览文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _is_likely_text(file_path: Path) -> bool:
    """简单判断文件是否为文本文件"""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(512)
        # 检查是否包含空字节（二进制文件特征）
        if b'\x00' in chunk:
            return False
        # 检查可打印字符比例
        text_chars = bytes(range(32, 127)) + b'\n\r\t\b'
        text_count = sum(c in text_chars for c in chunk)
        return text_count / len(chunk) > 0.7 if chunk else False
    except Exception:
        return False


@router.get("/server_info")
async def get_server_info(request: Request):
    """获取服务器信息，供前端动态配置API地址"""
    engine = request.app.state.engine
    # 返回完整的服务器信息
    return engine.settings.server_info


# ==================== 校对 API ====================

class ProofreadRequest(BaseModel):
    text: str = Field(..., description="待校对的文本")
    filename: Optional[str] = Field(None, description="文件名（可选，用于文档校对）")


class ProofreadResponse(BaseModel):
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="发现的问题列表")
    summary: Dict[str, Any] = Field(default_factory=dict, description="校对摘要")


# ==================== 文件解析 API ====================

@router.post("/parse-file", tags=["文件服务"])
async def parse_file(request: Request, file: UploadFile = File(...)):
    """
    解析上传的文件内容为纯文本
    支持: .txt, .md, .csv, .pdf, .docx, .doc
    """
    try:
        content = await parse_uploaded_file(file)
        return {
            "success": True,
            "filename": file.filename,
            "content": content,
            "content_length": len(content)
        }
    except Exception as e:
        logger.error(f"文件解析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proofread/text", tags=["校对服务"])
async def proofread_text(
    request: Request,
    body: ProofreadRequest
):
    """
    校对文本内容

    返回检测到的错别字、标点错误、用词不当等问题
    """
    engine = request.app.state.engine
    tenant_uuid = get_tenant_uuid_from_request(request)

    try:
        # 打印校对配置
        settings = engine.proofread_service.settings
        logger.info(f"[Proofread API] tenant={tenant_uuid}, PROOFREAD_USE_DEDICATED={getattr(settings, 'PROOFREAD_USE_DEDICATED', False)}, PROOFREAD_MODEL={getattr(settings, 'PROOFREAD_MODEL', '')}")
        service = engine.proofread_service
        result = await service.proofread_text(body.text)
        return result
    except Exception as e:
        logger.error(f"文本校对失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proofread/document", tags=["校对服务"])
async def proofread_document(
    request: Request,
    body: ProofreadRequest
):
    """
    校对文档内容

    支持 .txt, .doc, .docx, .pdf 等格式
    """
    engine = request.app.state.engine
    tenant_uuid = get_tenant_uuid_from_request(request)

    try:
        service = engine.proofread_service
        result = await service.proofread_document(body.text, body.filename)
        return result
    except Exception as e:
        logger.error(f"文档校对失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proofread/text/stream", tags=["校对服务"])
async def proofread_text_stream(
    request: Request,
    body: ProofreadRequest
):
    """
    校对文本内容（流式版本，实时返回进度）
    """
    engine = request.app.state.engine
    tenant_uuid = get_tenant_uuid_from_request(request)

    try:
        # 打印校对配置
        settings = engine.proofread_service.settings
        logger.info(f"[Proofread Stream API] tenant={tenant_uuid}, PROOFREAD_USE_DEDICATED={getattr(settings, 'PROOFREAD_USE_DEDICATED', False)}, PROOFREAD_MODEL={getattr(settings, 'PROOFREAD_MODEL', '')}")
        service = engine.proofread_service

        async def event_generator():
            all_issues = []
            total_batches = 0
            completed_batches = 0

            def progress_callback(current, total):
                nonlocal completed_batches
                completed_batches = current
                data = json.dumps({
                    "type": "progress",
                    "current": current,
                    "total": total
                })
                return f"data: {data}\n\n"

            # 先分块计算总数
            from src.services.proofread_service import split_paragraph_lst, BATCH_SIZE
            paragraphs = body.text.split('\n')
            chunks, _ = await split_paragraph_lst(paragraphs)
            # 规则引擎已禁用，避免与 LLM 重复
            rule_issues = []
            total_batches = max(1, (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE)

            # 发送初始信息
            yield f"data: {json.dumps({'type': 'start', 'total': total_batches, 'rule_issues': rule_issues})}\n\n"

            # 逐批处理以便实时推送
            from src.services.proofread_service import BATCH_SIZE as BS, MAX_CONCURRENCY
            import asyncio

            batches = []
            for i in range(0, len(chunks), BS):
                batches.append(chunks[i:i + BS])

            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

            # 检查是否使用中文纠错适配器
            use_corrector = service._init_corrector_adapter() and service._use_chinese_corrector
            logger.info(f"[Proofread Stream] use_corrector={use_corrector}")

            async def proofread_batch(batch):
                async with semaphore:
                    if use_corrector:
                        return await service._llm_proofread_batch_with_chinese_corrector(batch)
                    else:
                        return await service._llm_proofread_batch(batch)

            for i, batch in enumerate(batches):
                result = await proofread_batch(batch)
                completed_batches = i + 1
                if isinstance(result, list):
                    all_issues.extend(result)
                batch_data = json.dumps({
                    "type": "progress",
                    "current": completed_batches,
                    "total": total_batches,
                    "batch_issues": result
                })
                yield f"data: {batch_data}\n\n"

            # 合并规则引擎结果（已禁用）和去重
            all_issues.extend(rule_issues)
            all_issues = service._dedupe_issues(all_issues)
            summary = service._generate_summary(all_issues)

            # 打印返回给前端的完整数据（用于调试）
            logger.info(f"[Proofread Stream] 返回前端数据: issues_count={len(all_issues)}, summary={summary}")
            for idx, issue in enumerate(all_issues):
                logger.info(f"[Proofread Stream]   issue[{idx}]: {issue}")

            final_data = json.dumps({
                "type": "done",
                "issues": all_issues,
                "summary": summary,
                "reasoning": getattr(service, '_last_reasoning', '') or ''
            })
            yield f"data: {final_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"流式文本校对失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




