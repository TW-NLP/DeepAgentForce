"""
WebSocket 处理模块
支持实时流式对话 + 会话级别的对话历史管理
新增：完整保存思考过程（thinking_steps）
多租户支持：所有操作按 tenant_uuid 隔离
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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
        user_id = None  # 🆕 多租户 Skills 隔离（已废弃，统一使用 tenant_uuid）
        ws_connected = False  # 🆕 标记 WebSocket 是否已完成握手

        # 🆕 用于收集当前对话的思考过程
        current_thinking_steps = []
        token_buffer = ""
        token_flush_task: Optional[asyncio.Task] = None
        token_flush_interval = 0.03
        current_response_task: Optional[asyncio.Task] = None

        async def send_payload(event_type: str, data: Optional[dict] = None):
            """统一发送 WebSocket 事件，避免各处重复拼装 payload。"""
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"WS已断开，跳过发送: {event_type}")
                return

            payload = {
                "type": event_type,
                "data": data or {},
                "ts": datetime.now().isoformat(),
                "tenant_uuid": tenant_uuid,
            }
            json_str = json.dumps(payload, default=safe_json_serializer)
            await websocket.send_text(json_str)

        async def flush_token_buffer(force: bool = False):
            nonlocal token_buffer, token_flush_task
            if not token_buffer:
                if force:
                    token_flush_task = None
                return

            content = token_buffer
            token_buffer = ""
            token_flush_task = None
            await send_payload("token", {"content": content})

        async def schedule_token_flush():
            nonlocal token_flush_task
            if token_flush_task and not token_flush_task.done():
                return

            async def _delayed_flush():
                try:
                    await asyncio.sleep(token_flush_interval)
                    await flush_token_buffer(force=True)
                except asyncio.CancelledError:
                    return

            token_flush_task = asyncio.create_task(_delayed_flush())

        async def run_turn(message: str):
            nonlocal current_thinking_steps, session_id, agent, token_flush_task, token_buffer

            current_thinking_steps = []
            agent.status_callback = status_callback
            await send_payload("assistant_start", {"session_id": session_id})

            try:
                response = await agent.chat(message, thread_id=session_id)
            except asyncio.CancelledError:
                if token_flush_task and not token_flush_task.done():
                    token_flush_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await token_flush_task
                    token_flush_task = None
                if token_buffer:
                    await flush_token_buffer(force=True)
                await send_payload("assistant_stopped", {
                    "message": "已停止生成",
                    "session_id": session_id,
                })
                raise

            engine.history_manager.add_conversation(
                session_id=session_id,
                user_message=message,
                ai_response=response,
                thinking_steps=current_thinking_steps,
                metadata={"source": "websocket"},
                tenant_uuid=tenant_uuid
            )

            await ws_callback("done", {"message": response, "session_id": session_id})

        try:
            # 🆕 多租户：从 query_params 获取 token，验证并提取 tenant_uuid
            token = websocket.query_params.get("token")
            if token:
                tenant_uuid = extract_tenant_from_token(token)
                logger.info(f"WebSocket 连接，tenant_uuid: {tenant_uuid}")
            else:
                logger.warning("WebSocket 连接未携带 token，视为匿名会话")
            
            # 1. 定义极其强壮的回调函数
            status_callback = StatusCallback()
            
            async def ws_callback(event_type: str, data: dict):
                nonlocal token_buffer, token_flush_task
                # 🆕 收集思考过程
                if event_type == "step":
                    # 将每个 step 事件保存到当前对话的思考过程列表
                    current_thinking_steps.append({
                        "timestamp": datetime.now().isoformat(),
                        "event_type": event_type,
                        "data": data
                    })
                
                try:
                    if event_type == "token":
                        token_buffer += data.get("content", "")
                        # 达到一定长度立即 flush，否则做极短时间聚合
                        if len(token_buffer) >= 48:
                            if token_flush_task and not token_flush_task.done():
                                token_flush_task.cancel()
                                token_flush_task = None
                            await flush_token_buffer(force=True)
                        else:
                            await schedule_token_flush()
                        return

                    if token_buffer:
                        if token_flush_task and not token_flush_task.done():
                            token_flush_task.cancel()
                            token_flush_task = None
                        await flush_token_buffer(force=True)

                    await send_payload(event_type, data)
                except Exception as e:
                    # 捕获序列化错误，不要让它断开连接
                    logger.error(f"WS 发送失败 (序列化或网络问题): {str(e)}")

            status_callback.add_callback(ws_callback)
            
            # 2. 获取 Session（传入 tenant_uuid 用于多租户隔离）
            req_sid = websocket.query_params.get("session_id")
            session_id, agent = engine.get_or_create_session(
                session_id=req_sid,
                status_callback=status_callback,
                tenant_uuid=tenant_uuid,  # 🆕 传递 tenant_uuid
            )
            
            # 🆕 等待一段时间，确保客户端 onopen 已执行
            await asyncio.sleep(0.5)
            ws_connected = True
            
            # 3. 发送会话列表（按租户隔离）- 用于前端侧边栏显示
            try:
                all_sessions = engine.history_manager.list_sessions(tenant_uuid=tenant_uuid)
                logger.info(f"📋 找到 {len(all_sessions)} 个会话")
                if all_sessions:
                    # 发送轻量版会话列表给前端
                    session_list_payload = {
                        "type": "session_list",
                        "data": {
                            "sessions": [
                                {
                                    "session_id": s.get("session_id"),
                                    "title": s.get("title") or "新对话",
                                    "created_at": s.get("created_at"),
                                    "updated_at": s.get("updated_at"),
                                    "conversation_count": s.get("conversation_count", 0)
                                }
                                for s in all_sessions
                            ]
                        },
                        "ts": datetime.now().isoformat()
                    }
                    await send_payload("session_list", session_list_payload["data"])
                    logger.info(f"✅ 已发送会话列表到前端")
            except Exception as e:
                logger.error(f"发送会话列表失败: {e}", exc_info=True)

            # 4. 发送当前 session 的历史（如果有的话）
            history = engine.history_manager.get_session_history(session_id, tenant_uuid=tenant_uuid)
            if history:
                # 包装一下历史记录发送，防止这里也崩
                await ws_callback("history", history)

            # 4. 循环接收消息
            while True:
                if current_response_task is None:
                    data = await websocket.receive_json()
                else:
                    receive_task = asyncio.create_task(websocket.receive_json())
                    done, pending = await asyncio.wait(
                        {receive_task, current_response_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if current_response_task in done:
                        try:
                            await current_response_task
                        except asyncio.CancelledError:
                            pass
                        current_response_task = None

                        if not receive_task.done():
                            receive_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await receive_task
                        continue

                    data = receive_task.result()

                action = data.get("action", "message")

                if action == "stop":
                    if current_response_task is not None:
                        current_response_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await current_response_task
                        current_response_task = None
                    continue

                if current_response_task is not None:
                    await send_payload("error", {"message": "当前回答尚未完成，请先停止生成或等待结束。"})
                    continue

                message = data.get("message", "")
                if not message:
                    continue

                # 🆕 从请求体获取 session_id（前端在 JSON 中发送）
                incoming_sid = data.get("session_id")
                if incoming_sid and incoming_sid != session_id:
                    # 前端指定了 session_id，切换会话
                    session_id = incoming_sid
                    session_id, agent = engine.get_or_create_session(
                        session_id=session_id,
                        status_callback=status_callback,
                        tenant_uuid=tenant_uuid,
                    )

                current_response_task = asyncio.create_task(run_turn(message))

        except (WebSocketDisconnect, RuntimeError) as e:
            # 客户端主动断开或连接异常
            logger.info(f"WebSocket 连接结束: session={session_id}, tenant={tenant_uuid}, reason={type(e).__name__}")
            if current_response_task is not None:
                current_response_task.cancel()
            if session_id:
                try:
                    asyncio.get_running_loop().run_in_executor(
                        mining_executor, engine.user_preference.person_like_save, tenant_uuid, None
                    )
                except Exception as save_err:
                    logger.error(f"保存用户偏好失败: {save_err}")
        except Exception as e:
            logger.error(f"WS 主循环异常: {e}", exc_info=True)
            # 尝试发送错误给前端
            try:
                if current_response_task is not None:
                    current_response_task.cancel()
                if token_flush_task and not token_flush_task.done():
                    token_flush_task.cancel()
                if websocket.client_state == WebSocketState.CONNECTED:
                    await send_payload("error", {"message": str(e)})
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
