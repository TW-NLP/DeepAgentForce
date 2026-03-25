# src/workflow/callbacks.py
import asyncio

class EventType:
    STEP = "step"
    TOKEN = "token"
    ERROR = "error"

class StatusCallback:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    async def _emit(self, event_type: str, data: dict):
        for cb in self.callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(event_type, data)
            else:
                cb(event_type, data)

    # === 下面这些方法对应 conversational_agent.py 中的调用 ===
    
    async def on_agent_start(self, data: dict):
        # 对应前端图标：🤔
        await self._emit(EventType.STEP, {
            "step": "init",
            "title": "开始处理",
            "description": "正在解析用户意图..."
        })

    async def on_tool_start(self, data: dict):
        # 对应前端图标：🔧
        name = data.get("name", "Unknown Tool")
        args = str(data.get("args", ""))[:50]
        await self._emit(EventType.STEP, {
            "step": "tool_start",
            "title": f"调用工具: {name}",
            "description": f"参数: {args}"
        })

    async def on_tool_end(self, data: dict):
        # 对应前端图标：✅
        output = data.get("output", "")
        await self._emit(EventType.STEP, {
            "step": "tool_end",
            "title": "执行完成",
            "description": f"结果: {output}"
        })

    async def on_agent_finish(self, data: dict):
        # 对应前端图标：🎯
        await self._emit(EventType.STEP, {
            "step": "finish",
            "title": "处理结束",
            "description": "回答已生成"
        })

    async def on_token(self, token: str):
        """流式发送单个 token"""
        await self._emit(EventType.TOKEN, {"content": token})

    async def on_error(self, data: dict):
        await self._emit(EventType.ERROR, data)