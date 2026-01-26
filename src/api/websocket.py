"""
WebSocket 处理模块
支持实时流式对话
"""

import logging
import uuid
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
from typing import Dict

from src.api.routes import save_conversation_to_file
from src.workflow.agent import ConversationalAgent
from src.workflow.callbacks import StatusCallback, EventType

logger = logging.getLogger(__name__)


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
                    try:
                        # 建议把 save_conversation_to_file 放在这里调用
                        save_conversation_to_file(message, response) 
                        logger.info(f"WebSocket 对话已保存")
                    except Exception as save_error:
                        logger.error(f"保存历史失败: {save_error}")
                    # 发送完成信号
                    await websocket.send_json({
                        "type": "done",
                        "data": {
                            "message": response,
                            "status": "处理完成"
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"处理消息失败: {e}", exc_info=True)
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
        """
        await websocket.accept()
        session_id = None
        
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
                    try:
                        # 如果是异步函数加 await，如果是同步函数直接调用
                        # 建议把 save_conversation_to_file 放在这里调用
                        save_conversation_to_file(message, response) 
                        logger.info(f"WebSocket 对话已保存")
                    except Exception as save_error:
                        logger.error(f"保存历史失败: {save_error}")
                    
                    # 发送完成信号
                    await websocket.send_json({
                        "type": "done",
                        "message": response
                    })
                    
                except Exception as e:
                    logger.error(f"处理消息失败: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket 断开连接: {session_id}")
            if session_id:
                ws_session_manager.delete_session(session_id)
        except Exception as e:
            logger.error(f"WebSocket 错误: {str(e)}", exc_info=True)
            if session_id:
                ws_session_manager.delete_session(session_id)