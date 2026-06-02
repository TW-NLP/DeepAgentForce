"""从一个 .py 文件导入并抽取 LangChain 工具 —— 仅在**子进程**内执行。

这段逻辑原本在 ``CustomToolManager._load_tools_from_file`` 里直接跑在主进程，
现已迁出，只由 :mod:`tool_worker` 在受限子进程中调用，主进程不再 import 用户代码。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


def load_tools_from_file(path: Path) -> List[Any]:
    """从一个 .py 文件导入并抽取 LangChain 工具（在子进程内运行）。

    约定：
    - 若模块显式定义 ``TOOLS = [...]`` 列表，则取其中的 ``BaseTool``；
    - 否则把每个不以 ``_`` 开头、且带 docstring 的顶层函数自动包成 ``StructuredTool``。
    """
    from langchain_core.tools import BaseTool, StructuredTool

    path = Path(path)
    # 用显式 SourceFileLoader，避免依赖文件扩展名（临时校验文件可能是 .py.tmp）。
    mod_name = f"_custom_tool_{path.stem}_{int(time.time() * 1000)}"
    loader = SourceFileLoader(mod_name, str(path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    if spec is None:
        raise ImportError(f"无法为 {path} 创建模块 spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        loader.exec_module(module)
    finally:
        sys.modules.pop(mod_name, None)

    # 1) 显式 TOOLS
    explicit = getattr(module, "TOOLS", None)
    tools: List[Any] = []
    if isinstance(explicit, (list, tuple)):
        for t in explicit:
            if isinstance(t, BaseTool):
                tools.append(t)

    # 2) 自动包装顶层函数
    if not tools:
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            obj = getattr(module, attr)
            if callable(obj) and getattr(obj, "__module__", None) == mod_name:
                if obj.__doc__:
                    try:
                        tools.append(
                            StructuredTool.from_function(
                                func=obj,
                                name=attr,
                                description=obj.__doc__.strip().split("\n")[0],
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("包装函数 %s 失败: %s", attr, e)
    return tools


def tool_specs(path: Path) -> List[dict]:
    """抽取每个工具的元信息（name/description/参数 schema），供主进程重建代理工具。"""
    specs: List[dict] = []
    for t in load_tools_from_file(path):
        schema: dict = {}
        required: List[str] = []
        try:
            schema = t.args  # langchain: dict[name -> json schema]
        except Exception:  # noqa: BLE001
            schema = {}
        # 拿到完整 json schema 以区分必填/可选
        try:
            full = t.get_input_schema().model_json_schema()
            required = list(full.get("required", []) or [])
        except Exception:  # noqa: BLE001
            required = []
        specs.append(
            {
                "name": t.name,
                "description": t.description or "",
                "args": schema,
                "required": required,
            }
        )
    return specs
