"""
WebSocket 处理模块
支持实时流式对话 + 会话级别的对话历史管理
新增：完整保存思考过程（thinking_steps）
多租户支持：所有操作按 tenant_uuid 隔离
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
import jwt
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
from fastapi.websockets import WebSocketState
from src.workflow.callbacks import StatusCallback, EventType
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 线程池用于处理耗时的偏好挖掘任务，避免阻塞事件循环
mining_executor = ThreadPoolExecutor(max_workers=1)


def safe_json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "dict"): # 处理 Pydantic
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj) # 其他一切转字符串，保命要紧


def extract_tenant_from_token(token: str) -> Optional[str]:
    """
    从 JWT token 中提取 tenant_uuid
    用于 WebSocket 连接时的租户隔离验证
    """
    try:
        if not token:
            return None
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload.get("tenant_uuid")  # 🆕 使用 tenant_uuid
    except Exception as e:
        logger.warning(f"从 Token 提取 tenant_uuid 失败: {e}")
        return None

def setup_websocket_routes(app: FastAPI):
    engine = app.state.engine

    @app.websocket("/ws/stream")
    async def websocket_stream_simple(websocket: WebSocket):
        await websocket.accept()
        session_id = None
        tenant_uuid = None  # 🆕 多租户：当前会话所属租户
        
        # 🆕 用于收集当前对话的思考过程
        current_thinking_steps = []
        
        try:
            # 🆕 多租户：从 query_params 获取 token，验证并提取 tenant_uuid
            token = websocket.query_params.get("token")
            if token:
                tenant_uuid = extract_tenant_from_token(token)
                logger.info(f"WebSocket 连接，提取的 tenant_uuid: {tenant_uuid}")
            else:
                logger.warning("WebSocket 连接未携带 token，视为匿名会话")
            
            # 1. 定义极其强壮的回调函数
            status_callback = StatusCallback()
            
            async def ws_callback(event_type: str, data: dict):
                # 🆕 收集思考过程
                if event_type == "step":
                    # 将每个 step 事件保存到当前对话的思考过程列表
                    current_thinking_steps.append({
                        "timestamp": datetime.now().isoformat(),
                        "event_type": event_type,
                        "data": data
                    })
                
                # 关键检查：如果连接已关闭，不要尝试发送
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.warning(f"WS已断开，跳过发送: {event_type}")
                    return

                try:
                    # 构造消息
                    payload = {
                        "type": event_type, 
                        "data": data, 
                        "ts": datetime.now().isoformat(),
                        "tenant_uuid": tenant_uuid  # 🆕 前端可据此验证租户
                    }
                    
                    # 使用自定义序列化，防止报错
                    json_str = json.dumps(payload, default=safe_json_serializer)
                    
                    # 发送文本而不是 send_json，控制权更强
                    await websocket.send_text(json_str)
                    
                except Exception as e:
                    # 捕获序列化错误，不要让它断开连接
                    logger.error(f"WS 发送失败 (序列化或网络问题): {str(e)}")

            status_callback.add_callback(ws_callback)
            
            # 2. 获取 Session（传入 tenant_uuid 用于租户隔离）
            req_sid = websocket.query_params.get("session_id")
            session_id, agent = engine.get_or_create_session(
                session_id=req_sid, 
                status_callback=status_callback,
                tenant_uuid=tenant_uuid  # 🆕 传递 tenant_uuid
            )
            
            # 3. 发送历史记录（按租户隔离）
            history = engine.history_manager.get_session_history(session_id, tenant_uuid=tenant_uuid)
            if history:
                # 包装一下历史记录发送，防止这里也崩
                await ws_callback("history", history)

            # 4. 循环接收消息
            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")
                if not message: continue
                
                # 🆕 从请求体获取 session_id（前端在 JSON 中发送）
                incoming_sid = data.get("session_id")
                if incoming_sid and incoming_sid != session_id:
                    # 前端指定了 session_id，切换会话
                    session_id = incoming_sid
                    session_id, agent = engine.get_or_create_session(
                        session_id=session_id,
                        status_callback=status_callback,
                        tenant_uuid=tenant_uuid  # 🆕 保持 tenant_uuid
                    )
                
                # 🆕 重置思考过程列表（每次新对话开始时清空）
                current_thinking_steps = []
                
                # 双重保险：确保 agent 里的 callback 是最新的
                agent.status_callback = status_callback
                
                # 执行对话
                # 注意："666" 这种闲聊不会触发 tool_start，所以没有思考框是正常的
                # 要测试思考框，请问 "查询一下目前DeepSeek的消息"
                response = await agent.chat(message, thread_id=session_id)
                
                # 🆕 保存历史（包含思考过程，按租户隔离）
                engine.history_manager.add_conversation(
                    session_id=session_id, 
                    user_message=message, 
                    ai_response=response,
                    thinking_steps=current_thinking_steps,  # 传递思考过程
                    metadata={"source": "websocket"},
                    tenant_uuid=tenant_uuid  # 🆕 租户隔离
                )
                
                # 🆕 发送结束信号（包含 session_id）
                await ws_callback("done", {"message": response, "session_id": session_id})
        
        except WebSocketDisconnect:
            logger.info(f"客户端断开连接: session={session_id}, tenant={tenant_uuid}")
            if session_id:
                asyncio.get_running_loop().run_in_executor(
                    mining_executor, engine.user_preference.person_like_save, tenant_uuid, None
                )
        except Exception as e:
            logger.error(f"WS 主循环异常: {e}", exc_info=True)
            # 尝试发送错误给前端
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    pass

class ConversationHistoryManager:
    """
    会话级别的对话历史管理器
    多租户支持：每个租户有独立的历史目录和数据
    """
    
    def __init__(self, history_base_dir: Path):
        self.history_base_dir = Path(history_base_dir)
        self.history_base_dir.mkdir(parents=True, exist_ok=True)
        # 🆕 内存中的会话列表也按租户隔离
        self._memory_sessions: dict[str, dict] = {}  # key = f"{tenant_uuid}_{session_id}"

    def _make_key(self, session_id: str, tenant_uuid: Optional[str] = None) -> str:
        """生成内存中的唯一键"""
        if tenant_uuid:
            return f"{tenant_uuid}_{session_id}"
        return session_id

    def _get_tenant_dir(self, tenant_uuid: Optional[str] = None) -> Path:
        """获取租户专属的历史目录"""
        if tenant_uuid is None:
            tenant_uuid = "default"
        tenant_dir = self.history_base_dir / str(tenant_uuid)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir

    def _get_session_file_path(self, session_id: str, tenant_uuid: Optional[str] = None) -> Path:
        """获取会话文件的完整路径"""
        tenant_dir = self._get_tenant_dir(tenant_uuid)
        return tenant_dir / f"session_{session_id}.json"

    def create_session(self, session_id: str, tenant_uuid: Optional[str] = None) -> str:
        """创建新会话"""
        session_file = self._get_session_file_path(session_id, tenant_uuid)
        if session_file.exists():
            return session_id

        session_data = {
            "session_id": session_id,
            "tenant_uuid": tenant_uuid,  # 🆕 记录所属租户
            "title": "新对话",  # 初始标题，后续会更新
            "created_at": datetime.now().isoformat(),
            "conversations": []
        }
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return session_id

    def _generate_title(self, user_message: str) -> str:
        """根据用户消息生成标题摘要"""
        if not user_message:
            return "新对话"
        # 移除换行符，取前30个字符
        clean_msg = user_message.replace('\n', ' ').strip()
        if len(clean_msg) > 30:
            return clean_msg[:30] + '...'
        return clean_msg
    
    def add_conversation(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        thinking_steps: Optional[list] = None,  # 🆕 新增参数
        metadata: Optional[dict] = None,
        tenant_uuid: Optional[str] = None  # 🆕 多租户
    ):
        """
        添加对话记录，包含思考过程

        Args:
            session_id: 会话ID
            user_message: 用户消息
            ai_response: AI回复
            thinking_steps: 思考过程列表（包含所有 step 事件）
            metadata: 元数据
            tenant_uuid: 租户UUID 🆕
        """
        session_file = self._get_session_file_path(session_id, tenant_uuid)
        if not session_file.exists():
            self.create_session(session_id, tenant_uuid)

        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_content": user_message,
            "ai_content": ai_response
        }

        # 🆕 添加思考过程
        if thinking_steps:
            conversation["thinking_steps"] = thinking_steps

        if metadata:
            conversation["metadata"] = metadata

        session_data["conversations"].append(conversation)
        session_data["updated_at"] = datetime.now().isoformat()

        # 🆕 如果是第一条用户消息，更新会话标题
        if len(session_data["conversations"]) == 1:
            session_data["title"] = self._generate_title(user_message)

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def get_session_history(self, session_id: str, tenant_uuid: Optional[str] = None) -> Optional[dict]:
        """获取指定会话的历史"""
        session_file = self._get_session_file_path(session_id, tenant_uuid)
        if not session_file.exists():
            return None
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_sessions(self, tenant_uuid: Optional[str] = None) -> list:
        """
        列出指定租户的所有会话
        🆕 多租户：只返回当前租户的会话
        """
        tenant_dir = self._get_tenant_dir(tenant_uuid)
        sessions = []
        for session_file in tenant_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 🆕 安全检查：只返回属于该租户的会话
                    file_tenant = data.get("tenant_uuid")
                    if tenant_uuid is not None and file_tenant != tenant_uuid:
                        continue
                    sessions.append({
                        "session_id": data["session_id"],
                        "tenant_uuid": file_tenant,  # 🆕
                        "title": data.get("title", "历史对话"),  # 🆕 返回标题
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "conversation_count": len(data.get("conversations", [])),
                        "conversation": data.get("conversations", [])
                    })
            except Exception as e:
                logger.error(f"Failed to read session {session_file}: {e}")
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_session(self, session_id: str, tenant_uuid: Optional[str] = None) -> bool:
        """
        删除指定会话（按租户隔离）

        Args:
            session_id: 要删除的会话ID
            tenant_uuid: 租户UUID 🆕

        Returns:
            True 表示删除成功，False 表示会话不存在或删除失败
        """
        session_file = self._get_session_file_path(session_id, tenant_uuid)
        if session_file.exists():
            try:
                session_file.unlink()
                logger.info(f"已删除会话: {session_id} (tenant={tenant_uuid})")
                return True
            except Exception as e:
                logger.error(f"删除会话失败 {session_id}: {e}")
                return False
        return False