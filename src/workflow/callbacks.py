# src/workflow/callbacks.py
import asyncio


class EventType:
    STEP = "step"
    TOKEN = "token"
    ERROR = "error"
    ANSWER_START = "answer_start"


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

    async def on_agent_start(self, data: dict):
        await self._emit(EventType.STEP, {
            "step": "init",
            "title": "开始处理",
            "description": ""
        })

    async def on_tool_start(self, data: dict):
        name = data.get("name", "Unknown Tool")
        await self._emit(EventType.STEP, {
            "step": "tool_start",
            "title": f"执行 {name}",
            "description": ""
        })

    async def on_tool_end(self, data: dict):
        # 工具输出不进入 StepUpdate。
        # 工具返回结果通过 LLM 流式输出自然呈现给用户。
        # 思考区只展示步骤标题，不暴露原始输出。
        await self._emit(EventType.STEP, {
            "step": "tool_end",
            "title": "执行完成",
            "description": ""
        })

    async def on_agent_finish(self, data: dict):
        await self._emit(EventType.STEP, {
            "step": "finish",
            "title": "处理结束",
            "description": ""
        })

    async def on_agent_summarize(self, data: dict):
        """🆕 Agent 即将开始流式输出答案时的节点"""
        await self._emit(EventType.STEP, {
            "step": "summarize",
            "title": "正在生成答案",
            "description": ""
        })

    async def on_answer_start(self, data: dict):
        await self._emit(EventType.ANSWER_START, data or {})

    async def on_token(self, token: str):
        await self._emit(EventType.TOKEN, {"content": token})

    async def on_error(self, data: dict):
        await self._emit(EventType.ERROR, data)
