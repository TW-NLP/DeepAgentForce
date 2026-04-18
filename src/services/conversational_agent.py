import logging
from pathlib import Path
from typing import Optional, Any
import json
import subprocess
import os

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_community.tools import ShellTool
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from pydantic import Field

from src.services.person_like_service import UserPreferenceMining
from src.workflow.callbacks import StatusCallback
from config.settings import get_settings
from src.utils.setting_utils import load_config_from_file

logger = logging.getLogger(__name__)


def get_tenant_settings(tenant_uuid: Optional[str] = None) -> Any:
    """获取租户专属的设置对象"""
    settings = get_settings()
    if not tenant_uuid:
        return settings

    tenant_config = load_config_from_file(tenant_uuid)
    flat_config = tenant_config if isinstance(tenant_config, dict) else {}

    field_map = [
        "LLM_API_KEY", "LLM_URL", "LLM_MODEL",
        "EMBEDDING_API_KEY", "EMBEDDING_URL", "EMBEDDING_MODEL",
        "TAVILY_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_URL",
        "PROOFREAD_USE_DEDICATED", "PROOFREAD_API_URL", "PROOFREAD_API_KEY", "PROOFREAD_MODEL",
    ]
    for key in field_map:
        if key in flat_config:
            setattr(settings, key, flat_config[key])

    return settings


class SafeShellTool(BaseTool):
    """
    包装 ShellTool，自动修复 LLM 输出 JSON 数组格式的命令。
    例如：'["python x.py"]' → 'python x.py'
    """
    name: str = "shell"
    description: str = ""
    inner_tool: Any = Field(default=None)

    def _run(self, commands: Any) -> str:
        commands = self._sanitize(commands)
        logger.debug(f"[SafeShellTool] 执行命令: {commands}")
        return self.inner_tool._run(commands)

    async def _arun(self, commands: Any) -> str:
        commands = self._sanitize(commands)
        logger.debug(f"[SafeShellTool] 异步执行命令: {commands}")
        return await self.inner_tool._arun(commands)

    @staticmethod
    def _sanitize(commands: Any) -> str:
        """将 JSON 数组格式命令还原为纯字符串"""
        if not isinstance(commands, str):
            return str(commands)
        stripped = commands.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and parsed:
                    # 单命令直接取出，多命令用 && 连接
                    return " && ".join(str(c) for c in parsed)
            except json.JSONDecodeError:
                pass
        return commands


class ConversationalAgent:
    def __init__(
        self,
        settings,
        status_callback: Optional[StatusCallback] = None,
        tenant_uuid: Optional[str] = None,
    ):
        self.settings = settings
        self.status_callback = status_callback
        self.tenant_uuid = tenant_uuid

        # 🆕 多租户 Skills 目录
        self.builtin_skills_dir = Path(settings.SKILL_DIR)  # src/services/skills/
        self.user_skills_dir = Path(settings.USER_SKILL_DIR)  # data/skill/

        # 🆕 首次初始化：复制内置 Skills 到用户目录
        if self.tenant_uuid:
            from src.services.skill_manager import SkillManager
            skill_manager = SkillManager(self.builtin_skills_dir, self.user_skills_dir)
            skill_manager.initialize_tenant_skills(self.tenant_uuid)

        self.workspace = self._build_skills_workspace()

        self.user_profile_data = UserPreferenceMining(settings).get_frontend_format(
            tenant_uuid=tenant_uuid
        )
        self.user_summary = self.user_profile_data.get("summary", "No specific preference.")
        self._instance = None

        # ✅ 项目根路径在初始化时解析一次，后续直接用
        self.project_root = self._resolve_project_root()
        logger.info(f"[ConversationalAgent] 项目根路径: {self.project_root}, tenant_uuid: {tenant_uuid}, workspace: {self.workspace}")

    def _build_skills_workspace(self) -> str:
        """构建多租户 Skills 工作空间"""
        paths = [str(self.builtin_skills_dir)]
        # 🆕 用户 Skills 目录（每个 tenant_uuid 一个子目录）
        if self.tenant_uuid:
            tenant_dir = self.user_skills_dir / self.tenant_uuid
            if tenant_dir.exists():
                paths.append(str(tenant_dir))
        return ":".join(paths)

    # ------------------------------------------------------------------
    # 路径解析（只在启动时执行一次，不让 Agent 自己 find）
    # ------------------------------------------------------------------
    def _resolve_project_root(self) -> str:
        # 1. 优先读环境变量
        env_root = os.environ.get("DEEPAGENTFORCE_ROOT", "")
        if env_root and Path(env_root).exists():
            return env_root

        # 2. 从当前文件向上找，定位包含 src/services/skills 的目录
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src" / "services" / "skills").exists():
                return str(parent)

        # 3. 最终 fallback：find（仅在 init 时执行一次）
        try:
            result = subprocess.run(
                ["find", "/Users", "/home", "-type", "d",
                 "-name", "DeepAgentForce", "-maxdepth", "8"],
                capture_output=True, text=True, timeout=15,
            )
            first = result.stdout.strip().split("\n")[0]
            if first:
                return first
        except Exception as e:
            logger.warning(f"find 查找项目根失败: {e}")

        return ""

    # ------------------------------------------------------------------
    # 构建 SafeShellTool（每次 build_instance 时调用，保持配置最新）
    # ------------------------------------------------------------------
    def _build_shell_tool(self) -> SafeShellTool:
        # 🆕 使用项目根路径构建 RAG 脚本路径
        rag_script = (
            f"{self.project_root}/src/services/skills/rag-query/scripts/query.py"
            if self.project_root else "<未找到项目根，请设置 DEEPAGENTFORCE_ROOT 环境变量>"
        )
        # 🆕 用户专属 Skills 目录
        tenant_skills_info = f"\n  用户专属 Skills: {self.user_skills_dir}" if self.user_skills_dir.exists() else ""

        description = (
            "执行 shell 命令。\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️  格式规则（违反将导致执行失败）：\n"
            "  ✅ 正确: python /absolute/path/script.py 参数\n"
            "  ❌ 错误: [\"python /absolute/path/script.py 参数\"]  ← JSON 数组格式\n"
            "  ❌ 错误: 相对路径如 src/services/...\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"RAG 查询命令（直接复制使用，替换 <问题>）：\n"
            f"  python {rag_script} \"<问题>\" --tenant-uuid {self.tenant_uuid}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📁 Skills 搜索路径：\n"
            f"  内置 Skills: {self.builtin_skills_dir}\n"
            f"  用户专属 Skills: {self.user_skills_dir or '(未配置)'}\n"
        )
        return SafeShellTool(inner_tool=ShellTool(), description=description)

    # ------------------------------------------------------------------
    # 单例
    # ------------------------------------------------------------------
    def get_instance(self):
        if self._instance is None:
            self._instance = self.build_instance()
        return self._instance

    # ------------------------------------------------------------------
    # 构建 Agent
    # ------------------------------------------------------------------
    def build_instance(self):
        # 加载租户配置
        if self.tenant_uuid:
            self.settings = get_tenant_settings(self.tenant_uuid)
        elif not self.settings.LLM_MODEL:
            self.settings = get_settings()

        logger.info(
            f"正在使用模型: {self.settings.LLM_MODEL} 构建 Agent (tenant={self.tenant_uuid})"
        )

        model = init_chat_model(
            model=self.settings.LLM_MODEL,
            model_provider="openai",
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
        )

        # ✅ 使用 SafeShellTool，不再覆盖回 ShellTool
        exec_tool = self._build_shell_tool()

        # ✅ 项目根路径直接注入 system prompt，Agent 无需自己 find
        # rag_cmd_example 会插入到 f"" 的 system_prompt 中，JSON 的 { } 必须转义
        if self.project_root:
            _query_cmd = f"python {self.project_root}/src/services/skills/rag-query/scripts/query.py"
            _query_args = f'"{self.tenant_uuid}"'
            rag_cmd_example = f"{_query_cmd} {_query_args}"
        else:
            rag_cmd_example = "（项目根路径未找到，请设置 DEEPAGENTFORCE_ROOT）"

        system_prompt = f"""你是一个精确执行的智能体。闲聊直接回答，需要技能时找到对应技能并执行。

## 关键规则
1. 读取技能的 SKILL.md，严格按照其中 Execution 部分的命令格式执行
2. 区分位置参数和 --flag 参数，不得自行修改
3. 命令必须是纯文本字符串，绝对禁止 JSON 数组格式

## 输出格式规则（最最重要）

### 绝对禁止在回答中出现以下内容：
- **任何 SKILL.md 的原文或片段**（包括 name、description、version、Execution 等所有字段）
- **任何工具的原始输出内容**（包括 JSON、API 返回结果、命令行输出）
- **任何文件路径**（如 `/Users/...`、`<DEEPAGENTFORCE_ROOT>`、`.py` 脚本路径）
- **任何 YAML frontmatter**（`---` 开头的内容）
- **任何代码块内容**（\`\`\` 包围的内容）
- **"正在使用 XX 工具"、"调用 XX 接口"、"执行 XX 命令"** 等描述
- **"根据搜索结果"、"查询到以下内容"、"返回了 N 个结果"** 等前缀
- **表格格式的参数说明**（如 `| Parameter | Required |` 这样的表格）

### 正确做法：
- **只输出对用户有价值的自然语言回答**
- 工具执行后，用自己的话总结结果，不要复制工具输出
- 如果工具没有返回有用结果，直接告诉用户"未找到相关信息"或给出建议
- 回答应该像 ChatGPT 一样：干净、自然、直接

### 回答示例（对比）：

❌ 错误示例：
```
成功从 /path/加载配置
SKILL.md 内容...
根据搜索结果 [{{"title": "天气", "snippet": "..."}}]
```

✅ 正确示例：
```
今天北京天气晴朗，气温 15-23°C，适合户外活动。
```

## 环境信息（已解析，直接使用，禁止再次 find）
- 项目根路径: `{self.project_root}`
- 当前租户 UUID: `{self.tenant_uuid}`
- 内置 Skills 目录: `{self.builtin_skills_dir}`
- 用户专属 Skills 目录: `{self.user_skills_dir or '(无)'}` 🆕

## 文件保存规则（必须遵守）
当用户要求保存文件时，必须保存到以下目录：
- **保存路径**: `{self.project_root}/data/outputs/{{tenant_uuid}}/`
- **tenant_uuid**: `{self.tenant_uuid}`
- 示例：保存到 `/Users/tianwei/paper/DeepAgentForce/data/outputs/{self.tenant_uuid}/作文.txt`

⚠️ **禁止** 保存到其他任何目录（如项目根目录），否则文件不会被前端下载列表显示。

## RAG 查询命令模板（直接使用）
{rag_cmd_example}

## Skills 隔离规则（重要）🆕
- **内置 Skills**：所有用户共享，来自 `{self.builtin_skills_dir}`
- **用户专属 Skills**：仅当前用户可见，来自 `{self.user_skills_dir or '(无)'}`
- Agent 会自动搜索这两个目录查找可用的 Skill
- ⚠️ 不得访问其他用户的 Skills 目录

## 用户上下文
{self.user_summary}
"""

        return create_deep_agent(
            model=model,
            backend=FilesystemBackend(root_dir=str(self.settings.PROJECT_ROOT)),
            skills=[str(self.workspace)],
            tools=[exec_tool],
            checkpointer=MemorySaver(),
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # 对话入口
    # ------------------------------------------------------------------
    async def chat(self, user_input: str, thread_id: str = "default_thread") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        agent_instance = self.get_instance()
        token_buffer = ""

        try:
            if self.status_callback:
                logger.info(f"[{thread_id}] 触发 on_agent_start...")
                await self.status_callback.on_agent_start({"input": user_input})

            logger.info(f"[{thread_id}] 开始 Agent 流式处理: {user_input[:50]}")

            async for event in agent_instance.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                chunk, metadata = event[0], event[1]

                node = metadata.get("langgraph_node") if hasattr(metadata, "get") else None

                # 🆕 检测是否即将开始流式输出答案（Agent 节点 + 有内容 + token_buffer 刚开始）
                has_content = hasattr(chunk, "content") and chunk.content
                if node == "agent" and has_content and token_buffer == "":
                    if self.status_callback:
                        await self.status_callback.on_agent_summarize({})

                if has_content:
                    token = chunk.content
                    if isinstance(token, str):
                        token_buffer += token
                        if self.status_callback:
                            await self.status_callback.on_token(token)
                    elif isinstance(token, list):
                        for item in token:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                token_buffer += text
                                if self.status_callback:
                                    await self.status_callback.on_token(text)

                # 🆕 工具调用状态
                if node == "agent" and hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if self.status_callback:
                            await self.status_callback.on_tool_start(
                                {"name": tc["name"], "args": tc["args"]}
                            )
                elif node == "tools" and self.status_callback and hasattr(chunk, "content"):
                    await self.status_callback.on_tool_end(
                        {"output": str(chunk.content)}
                    )

            if self.status_callback:
                await self.status_callback.on_agent_finish({"output": token_buffer})

            return token_buffer

        except Exception as e:
            logger.error(f"Chat 处理失败: {e}", exc_info=True)
            if self.status_callback:
                await self.status_callback.on_error({"message": str(e)})
            return f"系统错误: {str(e)}"