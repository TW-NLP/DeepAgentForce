#!/usr/bin/env python
"""真实 LLM 实测：用已配置的租户跑通 agent，观察 skills / tools 的实际调用。

它会用租户的真实配置（LLM、Tavily 等）构建 ConversationalAgent，发送若干
代表性问题，并打印每个问题：①模型实际调用了哪些工具 ②最终回答。

⚠️ 会发起真实 LLM / 联网请求（产生费用）。

用法：
    PY=/Users/tianwei/miniforge3/envs/agent/bin/python

    # 跑预置问题套件（默认租户取已配置的那个）
    $PY scripts/demo_live_agent.py

    # 指定租户
    $PY scripts/demo_live_agent.py --tenant ba48e0d9-fd50-449c-be2c-e3c773360c26

    # 单个自定义问题
    $PY scripts/demo_live_agent.py --ask "帮我算一下 (123+456)*7"

    # 交互式连续对话（同一会话上下文）
    $PY scripts/demo_live_agent.py --chat
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 已配置 LLM 的默认租户（data/saved_config_<uuid>.json 存在的那个）
DEFAULT_TENANT = "ba48e0d9-fd50-449c-be2c-e3c773360c26"

DEFAULT_SUITE = [
    "你好，你是谁？",                                  # 纯闲聊，预期不调用工具
    "现在几点了？今天星期几？",                          # 预期 get_datetime
    "帮我算一下 (123 + 456) * 7 等于多少",               # 预期 calculator
    "记住：我喜欢用 Python 写脚本",                       # 预期 memory_write
    "我之前说过我喜欢用什么语言写脚本？",                  # 预期 memory_search
]


def _make_recording_callback():
    """返回一个会记录工具调用名的 StatusCallback 子类实例。"""
    from src.workflow.callbacks import StatusCallback

    class _Recorder(StatusCallback):
        def __init__(self):
            super().__init__()
            self.tool_calls: list[dict] = []

        async def on_tool_start(self, data: dict):
            self.tool_calls.append({"name": data.get("name"), "args": data.get("args")})
            await super().on_tool_start(data)

    return _Recorder()


async def _ask_one(agent, recorder, prompt: str, thread_id: str) -> None:
    recorder.tool_calls.clear()
    print("\n" + "─" * 70)
    print(f"🧑 用户：{prompt}")
    answer = await agent.chat(prompt, thread_id=thread_id)
    if recorder.tool_calls:
        names = ", ".join(
            f"{c['name']}({_short_args(c['args'])})" for c in recorder.tool_calls
        )
        print(f"🛠  调用工具：{names}")
    else:
        print("🛠  调用工具：（无，直接回答）")
    print(f"🤖 回答：{answer}")


def _short_args(args) -> str:
    s = str(args)
    return s if len(s) <= 60 else s[:57] + "..."


async def run_suite(agent, recorder, prompts: list[str]) -> None:
    thread_id = "demo-suite"
    for p in prompts:
        await _ask_one(agent, recorder, p, thread_id)


async def run_chat(agent, recorder) -> None:
    print("进入交互模式（输入 exit / quit 退出）。同一会话共享上下文。")
    thread_id = "demo-chat"
    loop = asyncio.get_event_loop()
    while True:
        try:
            prompt = await loop.run_in_executor(None, input, "\n🧑 你：")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in {"exit", "quit", "q"}:
            break
        if not prompt.strip():
            continue
        await _ask_one(agent, recorder, prompt, thread_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="租户 UUID（其 saved_config 提供 LLM 配置）")
    parser.add_argument("--ask", help="只问一个自定义问题")
    parser.add_argument("--chat", action="store_true", help="交互式连续对话")
    args = parser.parse_args()

    from config.settings import get_settings
    from src.services.conversational_agent import ConversationalAgent

    settings = get_settings()
    recorder = _make_recording_callback()

    print(f"构建 agent（tenant={args.tenant}）…")
    agent = ConversationalAgent(
        settings=settings, status_callback=recorder, tenant_uuid=args.tenant
    )
    # 触发一次构建，确认模型配置可用
    agent.get_instance()
    print(f"✅ 模型：{agent.settings.LLM_MODEL}  base_url={agent.settings.LLM_BASE_URL}")

    if args.chat:
        asyncio.run(run_chat(agent, recorder))
    elif args.ask:
        asyncio.run(run_suite(agent, recorder, [args.ask]))
    else:
        asyncio.run(run_suite(agent, recorder, DEFAULT_SUITE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
