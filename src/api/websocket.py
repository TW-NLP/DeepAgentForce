"""
WebSocket 处理模块
支持实时流式对话 + 会话级别的对话历史管理
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
from typing import Dict
from config import settings
from src.services.person_like_service import UserPreferenceMining
from src.workflow.agent import ConversationalAgent
from src.workflow.callbacks import StatusCallback, EventType
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

mining_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MiningWorker")
# ==================== WebSocket Session 管理 ====================

class WebSocketSessionManager:
    """WebSocket Session 管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationalAgent] = {}
    
    def create_session(self, status_callback: StatusCallback) -> tuple[str, ConversationalAgent]:
        """创建新 Session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ConversationalAgent(status_callback)
        logger.info(f"创建 WebSocket 会话: {session_id}")
        return session_id, self.sessions[session_id]
    
    def get_session(self, session_id: str) -> ConversationalAgent:
        """获取 Session"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """删除 Session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除 WebSocket 会话: {session_id}")


# 全局 WebSocket Session 管理器
ws_session_manager = WebSocketSessionManager()


# ==================== WebSocket 路由设置 ====================

def setup_websocket_routes(app: FastAPI):
    """
    设置 WebSocket 路由
    
    Args:
        app: FastAPI 应用实例
    """
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket 流式对话接口
        支持真实的工作流状态回调和流式输出
        ✅ 新增：会话级别的对话历史管理
        """
        await websocket.accept()
        session_id = None
        
        try:
            # 创建状态回调
            status_callback = StatusCallback()
            
            # 定义回调函数：将状态发送到 WebSocket
            async def ws_callback(event_type: str, data: dict):
                """将工作流状态实时发送到前端"""
                try:
                    await websocket.send_json({
                        "type": event_type,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"发送 WebSocket 消息失败: {e}")
            
            # 注册回调
            status_callback.add_callback(ws_callback)
            
            # 创建 Session
            session_id, agent = ws_session_manager.create_session(status_callback)
            
            # 发送 Session ID
            await websocket.send_json({
                "type": "session",
                "data": {
                    "session_id": session_id,
                    "message": "连接成功"
                }
            })
            
            while True:
                # 接收消息
                data = await websocket.receive_json()
                message = data.get("message", "")
                
                if not message:
                    continue
                
                # 发送开始信号
                await websocket.send_json({
                    "type": "start",
                    "data": {
                        "message": message,
                        "status": "开始处理..."
                    }
                })
                
                try:
                    # 处理消息（工作流会自动通过回调发送状态）
                    logger.info(f"[{session_id}] 处理消息: {message[:50]}...")
                    response = await agent.chat(message)
                    
                    # ✅ 【关键修改】使用新的会话级别保存
                    try:
                        save_conversation_to_session(
                            session_id=session_id,
                            user_message=message,
                            ai_response=response,
                            metadata={
                                "endpoint": "/ws/chat",
                                "client_host": websocket.client.host if websocket.client else None,
                                "client_port": websocket.client.port if websocket.client else None,
                            }
                        )
                        logger.info(f"[{session_id}] 对话已保存到会话历史")
                    except Exception as save_error:
                        logger.error(f"[{session_id}] 保存对话历史失败: {save_error}", exc_info=True)
                    
                    # 发送完成信号
                    await websocket.send_json({
                        "type": "done",
                        "data": {
                            "message": response,
                            "status": "处理完成"
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"[{session_id}] 处理消息失败: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "status": "处理失败"
                        }
                    })
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket 断开连接: {session_id}")
            if session_id:
                ws_session_manager.delete_session(session_id)
                # 注意：不删除会话历史文件，保留记录
        except Exception as e:
            logger.error(f"WebSocket 错误: {str(e)}", exc_info=True)
            if session_id:
                ws_session_manager.delete_session(session_id)
            try:
                await websocket.close()
            except:
                pass
    
    @app.websocket("/ws/stream")
    async def websocket_stream_simple(websocket: WebSocket):
        """
        简化版 WebSocket 接口（向后兼容）
        每次连接创建新的对话，支持真实状态回调
        ✅ 新增：会话级别的对话历史管理
        """
        await websocket.accept()
        session_id = None

        def run_mining_bg_task(sid: str):
            try:
                logger.info(f"[{sid}] ⏳ 触发后台任务: 开始挖掘用户偏好...")
                # 实例化挖掘服务
                service = UserPreferenceMining()
                # 执行耗时操作
                service.person_like_save()
                logger.info(f"[{sid}] ✅ 后台任务完成: 用户画像已更新")
            except Exception as e:
                logger.error(f"[{sid}] ❌ 后台挖掘任务失败: {str(e)}", exc_info=True)
        
        try:
            # 创建状态回调
            status_callback = StatusCallback()
            
            # 定义回调函数
            async def ws_callback(event_type: str, data: dict):
                """将工作流状态实时发送到前端"""
                try:
                    if event_type == EventType.STEP:
                        await websocket.send_json({
                            "type": "step",
                            "step": data.get("step", ""),
                            "title": data.get("title", ""),
                            "description": data.get("description", "")
                        })
                    elif event_type == EventType.TOKEN:
                        await websocket.send_json({
                            "type": "token",
                            "content": data.get("content", "")
                        })
                    elif event_type == "progress":
                        await websocket.send_json({
                            "type": "progress",
                            "current": data.get("current", 0),
                            "total": data.get("total", 0),
                            "description": f"处理中 {data.get('current', 0)}/{data.get('total', 0)}"
                        })
                    elif event_type == EventType.ERROR:
                        await websocket.send_json({
                            "type": "error",
                            "message": data.get("message", "")
                        })
                except Exception as e:
                    logger.error(f"发送 WebSocket 消息失败: {e}")
            
            # 注册回调
            status_callback.add_callback(ws_callback)
            
            # 创建 Agent
            session_id, agent = ws_session_manager.create_session(status_callback)
            


            while True:
                # 接收消息
                data = await websocket.receive_json()
                message = data.get("message", "")

                
                
                if not message:
                    continue
                
                try:
                    # 处理消息
                    logger.info(f"[{session_id}] 处理消息: {message[:50]}...")
                    response = await agent.chat(message)
                    
                    # ✅ 【关键修改】使用新的会话级别保存
                    try:
                        save_conversation_to_session(
                            session_id=session_id,
                            user_message=message,
                            ai_response=response,
                            metadata={
                                "endpoint": "/ws/stream",
                                "client_host": websocket.client.host if websocket.client else None,
                                "client_port": websocket.client.port if websocket.client else None,
                            }
                        )
                        logger.info(f"[{session_id}] 对话已保存到会话历史")
                    except Exception as save_error:
                        logger.error(f"[{session_id}] 保存对话历史失败: {save_error}", exc_info=True)
                    
                    # 发送完成信号
                    await websocket.send_json({
                        "type": "done",
                        "message": response
                    })
                    
                except Exception as e:
                    logger.error(f"[{session_id}] 处理消息失败: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket 断开连接: {session_id}")
            if session_id:
                ws_session_manager.delete_session(session_id)
                try:
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(
                        mining_executor, 
                        run_mining_bg_task, 
                        session_id
                    )
                except Exception as e:
                    logger.error(f"无法启动后台挖掘任务: {e}")
        except Exception as e:
            logger.error(f"WebSocket 错误: {str(e)}", exc_info=True)
            if session_id:
                ws_session_manager.delete_session(session_id)


class ConversationHistoryManager:
    """会话级别的对话历史管理器"""
    
    def __init__(self, history_dir: str = settings.HISTORY_FILE):
        """
        初始化历史管理器
        
        Args:
            history_dir: 历史记录保存目录
        """
        self.history_dir = history_dir
        self.history_dir.mkdir(exist_ok=True)
        
    def _get_session_file_path(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return self.history_dir / f"session_{session_id}.json"
    
    def create_session(self, session_id: str) -> str:
        """
        创建新会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话ID
        """
        session_file = self._get_session_file_path(session_id)
        
        # 如果文件已存在，不覆盖
        if session_file.exists():
            return session_id
            
        # 创建空会话文件
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "conversations": []
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
            
        return session_id
    
    def add_conversation(
        self, 
        session_id: str, 
        user_message: str, 
        ai_response: str,
        metadata: Optional[dict] = None
    ):
        """
        添加对话到会话
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            ai_response: AI回复
            metadata: 额外的元数据（可选）
        """
        session_file = self._get_session_file_path(session_id)
        
        # 如果会话文件不存在，先创建
        if not session_file.exists():
            self.create_session(session_id)
        
        # 读取现有会话数据
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # 构建对话记录
        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_content": user_message,
            "ai_content": ai_response
        }
        
        # 添加元数据
        if metadata:
            conversation["metadata"] = metadata
        
        # 添加到会话
        session_data["conversations"].append(conversation)
        session_data["updated_at"] = datetime.now().isoformat()
        
        # 保存
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    def get_session_history(self, session_id: str) -> Optional[dict]:
        """
        获取会话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据，如果不存在返回None
        """
        session_file = self._get_session_file_path(session_id)
        
        if not session_file.exists():
            return None
            
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_sessions(self) -> list:
        """
        列出所有会话
        
        Returns:
            会话列表，包含会话ID和基本信息
        """
        sessions = []
        
        for session_file in self.history_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "conversation_count": len(data.get("conversations", [])),
                        "conversation": data.get("conversations", [])
                    })
            except Exception as e:
                print(f"读取会话文件失败 {session_file}: {e}")
                
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        session_file = self._get_session_file_path(session_id)
        
        if session_file.exists():
            session_file.unlink()
            return True
        return False
    
    def export_session_to_markdown(self, session_id: str, output_path: Optional[str] = None) -> str:
        """
        导出会话为Markdown格式
        
        Args:
            session_id: 会话ID
            output_path: 输出路径（可选）
            
        Returns:
            Markdown内容
        """
        session_data = self.get_session_history(session_id)
        
        if not session_data:
            return ""
        
        # 构建Markdown
        md_lines = [
            f"# 会话记录: {session_id}",
            f"",
            f"**创建时间**: {session_data.get('created_at', 'N/A')}  ",
            f"**更新时间**: {session_data.get('updated_at', 'N/A')}  ",
            f"**对话轮数**: {len(session_data.get('conversations', []))}",
            f"",
            "---",
            ""
        ]
        
        for idx, conv in enumerate(session_data.get("conversations", []), 1):
            md_lines.extend([
                f"## 对话 {idx}",
                f"",
                f"**时间**: {conv.get('timestamp', 'N/A')}  ",
                f"**ID**: `{conv.get('id', 'N/A')}`",
                f"",
                f"### 用户",
                f"",
                conv.get("user_content", ""),
                f"",
                f"### AI",
                f"",
                conv.get("ai_content", ""),
                f"",
                "---",
                ""
            ])
        
        md_content = "\n".join(md_lines)
        
        # 如果指定了输出路径，保存文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        return md_content


# 全局实例
history_manager = ConversationHistoryManager()


def save_conversation_to_session(
    session_id: str,
    user_message: str,
    ai_response: str,
    metadata: Optional[dict] = None
):
    """
    保存对话到会话（便捷函数）
    
    Args:
        session_id: 会话ID
        user_message: 用户消息
        ai_response: AI回复
        metadata: 额外的元数据（可选）
    """
    history_manager.add_conversation(
        session_id=session_id,
        user_message=user_message,
        ai_response=ai_response,
        metadata=metadata
    )