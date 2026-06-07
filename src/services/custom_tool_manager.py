"""自定义 Python 工具管理（多租户）。

用户可以上传 ``.py`` 文件作为自定义工具，按 ``tenant_uuid`` 隔离存放在
``DATA_DIR/agent_tools_custom/<tenant_uuid>/<tool_id>.py``。这些工具会进入
``ConversationalAgent._collect_extra_tools()`` 的「额外工具池」，和 MCP 工具一样
自动享受 tool_disclosure 的渐进式披露。

**约定（让用户写最少的样板）：**
- 文件里写普通函数即可：每个不以 ``_`` 开头、且带 docstring 的顶层函数，会被自动
  包装成一个 LangChain ``StructuredTool``（函数名=工具名，docstring=描述）。
- 高级用法：在文件里显式定义 ``TOOLS = [...]``（一组 ``BaseTool``），则以它为准。

⚠️ **安全说明**：导入用户上传的 Python 文件会**真实执行**该模块顶层代码。为此本模块
**从不在主进程导入用户代码**：所有「导入并抽取工具」(introspect) 与「调用工具」(invoke)
都委托给 :class:`src.services.sandbox.SandboxRunner` 派生的受限子进程（CPU/文件大小/
地址空间 rlimit + 墙钟超时 + 过滤密钥环境变量）。主进程只持有转发到子进程的代理工具。
另外保留基础静态扫描（危险调用给出 warning，不强制拦截）。
"""

from __future__ import annotations

import ast
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.sandbox import SandboxRunner, SandboxError
from src.services.tool_taxonomy import normalize_type

logger = logging.getLogger(__name__)


def _annotate_custom(tool: Any, type_slug: str) -> None:
    """把 Type 写进自定义工具的 metadata（tool_search 重排用，容错）。"""
    try:
        md = dict(getattr(tool, "metadata", None) or {})
        md["custom_type"] = type_slug or ""
        tool.metadata = md
    except Exception:  # noqa: BLE001 - metadata 只读则忽略
        pass

# 文件名/工具 id 允许的字符
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# 静态扫描时给出 warning 的「危险」标识（不强制拦截，仅提示）。
_RISKY_PATTERNS = [
    "os.system", "subprocess", "eval(", "exec(", "__import__",
    "shutil.rmtree", "socket.", "pickle.loads",
]

_TEMPLATE = '''"""示例自定义工具：把每个函数自动暴露为一个工具。

约定：
- 不以下划线开头、且带 docstring 的顶层函数会被自动注册为工具；
- 函数的类型注解用于生成参数 schema；docstring 即工具描述。
"""


def greet(name: str) -> str:
    """根据名字生成一句问候语。"""
    return f"你好，{name}！"


def add_numbers(a: float, b: float) -> float:
    """计算两个数字之和。"""
    return a + b
'''


class CustomToolManager:
    """管理某项目的自定义 Python 工具文件（多租户）。"""

    def __init__(self, base_dir: str | Path, runner: Optional[SandboxRunner] = None):
        # base_dir = DATA_DIR/agent_tools_custom
        self.base_dir = Path(base_dir)
        # 子进程沙箱：导入/调用用户代码都在此处隔离，主进程永不 import。
        self.runner = runner or SandboxRunner()

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    def _tenant_dir(self, tenant_uuid: Optional[str]) -> Path:
        tid = tenant_uuid or "default"
        d = self.base_dir / tid
        return d

    def _tool_path(self, tenant_uuid: Optional[str], tool_id: str) -> Path:
        return self._tenant_dir(tenant_uuid) / f"{tool_id}.py"

    @staticmethod
    def _valid_id(tool_id: str) -> bool:
        return bool(tool_id) and bool(_SAFE_ID_RE.match(tool_id))

    # ------------------------------------------------------------------
    # 元数据（Type 等，按租户存于 <tenant>/_meta.json，按 tool_id 索引）
    # ------------------------------------------------------------------
    def _meta_file(self, tenant_uuid: Optional[str]) -> Path:
        return self._tenant_dir(tenant_uuid) / "_meta.json"

    def _read_meta(self, tenant_uuid: Optional[str]) -> Dict[str, Dict[str, Any]]:
        path = self._meta_file(tenant_uuid)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - 元数据坏掉不应影响工具加载
            return {}

    def _write_meta(self, tenant_uuid: Optional[str], meta: Dict[str, Dict[str, Any]]) -> None:
        path = self._meta_file(tenant_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_tool_type(self, tenant_uuid: Optional[str], tool_id: str) -> str:
        return normalize_type(self._read_meta(tenant_uuid).get(tool_id, {}).get("type"))

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate_code(self, code: str) -> Dict[str, Any]:
        """静态校验：语法是否合法、能否抽出工具、是否含危险调用。

        返回 {valid, errors, warnings, tool_names}。不执行代码。
        """
        errors: List[str] = []
        warnings: List[str] = []
        tool_names: List[str] = []

        if not code or not code.strip():
            return {"valid": False, "errors": ["代码为空"], "warnings": [], "tool_names": []}

        # 1) 语法
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"语法错误：第 {e.lineno} 行 {e.msg}"],
                "warnings": [],
                "tool_names": [],
            }

        # 2) 抽取候选工具名（顶层函数 / TOOLS 变量）
        has_tools_var = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and ast.get_docstring(node):
                    tool_names.append(node.name)
                elif not node.name.startswith("_"):
                    warnings.append(f"函数 {node.name} 缺少 docstring，将被忽略")
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "TOOLS":
                        has_tools_var = True

        if not tool_names and not has_tools_var:
            errors.append("未发现可用工具：请定义带 docstring 的顶层函数，或定义 TOOLS 列表")

        # 3) 危险调用扫描（仅警告）
        for pat in _RISKY_PATTERNS:
            if pat in code:
                warnings.append(f"检测到潜在危险调用 `{pat}`，请确认其必要性与安全性")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "tool_names": tool_names,
        }

    # ------------------------------------------------------------------
    # 增删改查
    # ------------------------------------------------------------------
    def save_tool(
        self,
        tenant_uuid: Optional[str],
        tool_id: str,
        code: str,
        force: bool = False,
        tool_type: str = "",
    ) -> Dict[str, Any]:
        """保存（新增或覆盖）一个自定义工具文件。``tool_type`` 为固定类目表里的 Type。"""
        if tenant_uuid is None:
            return {"success": False, "message": "需要登录才能保存自定义工具"}
        tool_id = (tool_id or "").strip()
        if not self._valid_id(tool_id):
            return {"success": False, "message": "工具 id 只能包含字母、数字、下划线和连字符"}

        check = self.validate_code(code)
        if not check["valid"]:
            return {"success": False, "message": "代码校验失败", "errors": check["errors"]}

        path = self._tool_path(tenant_uuid, tool_id)
        if path.exists() and not force:
            return {"success": False, "message": f"工具 '{tool_id}' 已存在，使用覆盖保存以替换"}

        # 运行期校验：在沙箱子进程里导入一次，确认能产出工具
        try:
            self._tenant_dir(tenant_uuid).mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".py.tmp")
            tmp.write_text(code, encoding="utf-8")
            try:
                specs = self.runner.introspect(tmp)
            finally:
                tmp.unlink(missing_ok=True)
            if not specs:
                return {"success": False, "message": "导入成功但未产出任何工具"}
        except SandboxError as e:
            return {"success": False, "message": f"导入失败：{e}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"导入失败：{e}"}

        path.write_text(code, encoding="utf-8")

        # 记录 Type（归一化到固定类目表，非法/缺失则记空 Type）。
        meta = self._read_meta(tenant_uuid)
        meta[tool_id] = {"type": normalize_type(tool_type)}
        self._write_meta(tenant_uuid, meta)

        return {
            "success": True,
            "message": f"已保存自定义工具 '{tool_id}'（含 {len(specs)} 个工具）",
            "tool_id": tool_id,
            "tool_names": [s["name"] for s in specs],
            "type": meta[tool_id]["type"],
            "warnings": check.get("warnings", []),
        }

    def delete_tool(self, tenant_uuid: Optional[str], tool_id: str) -> Dict[str, Any]:
        if tenant_uuid is None:
            return {"success": False, "message": "需要登录才能删除自定义工具"}
        if not self._valid_id(tool_id):
            return {"success": False, "message": "非法工具 id"}
        path = self._tool_path(tenant_uuid, tool_id)
        if not path.exists():
            return {"success": False, "message": f"工具 '{tool_id}' 不存在"}
        path.unlink()
        meta = self._read_meta(tenant_uuid)
        if tool_id in meta:
            meta.pop(tool_id)
            self._write_meta(tenant_uuid, meta)
        return {"success": True, "message": f"已删除自定义工具 '{tool_id}'"}

    def get_tool_code(self, tenant_uuid: Optional[str], tool_id: str) -> Optional[str]:
        if not self._valid_id(tool_id):
            return None
        path = self._tool_path(tenant_uuid, tool_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_tools(self, tenant_uuid: Optional[str]) -> List[Dict[str, Any]]:
        """列出该租户的自定义工具文件（含其暴露的工具名）。"""
        d = self._tenant_dir(tenant_uuid)
        if not d.exists():
            return []
        meta = self._read_meta(tenant_uuid)
        out: List[Dict[str, Any]] = []
        for path in sorted(d.glob("*.py")):
            tool_id = path.stem
            stat = path.stat()
            entry: Dict[str, Any] = {
                "tool_id": tool_id,
                "source": "custom",
                "type": normalize_type(meta.get(tool_id, {}).get("type")),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "tools": [],
                "error": None,
            }
            try:
                specs = self.runner.introspect(path)
                entry["tools"] = [
                    {"name": s["name"], "description": s.get("description") or ""}
                    for s in specs
                ]
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # 加载（供 Agent 注入）
    # ------------------------------------------------------------------
    def load_all_tools(self, tenant_uuid: Optional[str]) -> List[Any]:
        """加载该租户全部自定义工具的**代理**（实际调用转发到沙箱子进程）。

        单个文件失败不影响其余；主进程不会 import 用户代码。
        """
        d = self._tenant_dir(tenant_uuid)
        if not d.exists():
            return []
        meta = self._read_meta(tenant_uuid)
        all_tools: List[Any] = []
        for path in sorted(d.glob("*.py")):
            try:
                proxies = self.runner.build_proxy_tools(path)
                ttype = normalize_type(meta.get(path.stem, {}).get("type"))
                for t in proxies:
                    _annotate_custom(t, ttype)
                all_tools.extend(proxies)
            except Exception as e:  # noqa: BLE001
                logger.warning("[CustomTool] 加载 %s 失败，跳过: %s", path.name, e)
        return all_tools

    def get_template(self) -> str:
        return _TEMPLATE
