"""主进程侧：把「导入/调用自定义工具」隔离到受限子进程，并暴露代理工具。

:class:`SandboxRunner`：
- ``introspect(path)`` → 子进程导入文件，返回 [{name, description, args}]。
- ``invoke(path, name, kwargs)`` → 子进程执行某工具，返回字符串结果（永不抛）。
- ``build_proxy_tools(path)`` → 先 introspect，再为每个工具造一个 ``StructuredTool``
  代理（args_schema 由 introspect 的 json schema 重建），其 ``func``/``coroutine`` 把
  调用转发回 ``invoke``。主进程因此从不 import 用户代码。

隔离手段（POSIX）：``preexec_fn`` 里 ``setrlimit`` 限制 CPU 秒/输出文件大小/地址空间
（RLIMIT_AS 仅 Linux），加墙钟超时、过滤掉密钥类环境变量、独立临时 cwd、
``start_new_session=True`` 便于整组超时清理。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKER = Path(__file__).resolve().parent / "tool_worker.py"

# 过滤掉名字里带这些片段的环境变量（避免把主进程密钥泄露给用户代码）。
_SECRET_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API)", re.IGNORECASE)

# 资源上限（可调）。
_CPU_SECONDS = 10          # CPU 时间
_WALL_SECONDS = 20         # 墙钟超时
_FSIZE_BYTES = 16 * 1024 * 1024     # 单个输出文件最大 16MB
_AS_BYTES = 1024 * 1024 * 1024      # 地址空间 1GB（仅 Linux 生效）


class SandboxError(Exception):
    """沙箱执行失败（导入/超时/崩溃）。"""


def _filtered_env() -> Dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not _SECRET_RE.search(k)
    }
    # 保底的运行环境
    env.setdefault("PATH", os.environ.get("PATH", ""))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 让子进程内的 import 找得到项目（worker 自己也会 insert，这里冗余兜底）
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    return env


def _preexec():  # pragma: no cover - 只在 POSIX 子进程里跑
    """子进程 exec 前设置 rlimit。仅 POSIX 可用。"""
    import resource

    def _set(res, soft):
        try:
            hard = resource.getrlimit(res)[1]
            cap = soft if hard == resource.RLIM_INFINITY else min(soft, hard)
            resource.setrlimit(res, (cap, hard))
        except Exception:
            pass

    _set(resource.RLIMIT_CPU, _CPU_SECONDS)
    _set(resource.RLIMIT_FSIZE, _FSIZE_BYTES)
    # RLIMIT_AS 在 macOS 上会误伤（Python 解释器本身就要映射很多内存），仅 Linux 设。
    if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
        _set(resource.RLIMIT_AS, _AS_BYTES)


class SandboxRunner:
    """派生受限子进程来导入/调用自定义工具。"""

    def __init__(
        self,
        python_executable: Optional[str] = None,
        wall_seconds: int = _WALL_SECONDS,
    ):
        self.python = python_executable or sys.executable
        self.wall_seconds = wall_seconds

    # ------------------------------------------------------------------
    # 底层：跑一次 worker
    # ------------------------------------------------------------------
    def _run(self, mode: str, file_path: Path, request: Optional[dict]) -> dict:
        file_path = Path(file_path).resolve()
        with tempfile.TemporaryDirectory(prefix="sbx_") as tmp:
            tmp_dir = Path(tmp)
            req_path = tmp_dir / "request.json"
            resp_path = tmp_dir / "response.json"
            req_path.write_text(json.dumps(request or {}, ensure_ascii=False), encoding="utf-8")

            cmd = [self.python, str(_WORKER), mode, str(file_path), str(req_path), str(resp_path)]
            popen_kwargs: Dict[str, Any] = dict(
                cwd=str(tmp_dir),
                env=_filtered_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if os.name == "posix":
                popen_kwargs["preexec_fn"] = _preexec
                popen_kwargs["start_new_session"] = True

            try:
                proc = subprocess.run(
                    cmd,
                    timeout=self.wall_seconds,
                    **popen_kwargs,
                )
            except subprocess.TimeoutExpired:
                raise SandboxError(f"子进程超时（>{self.wall_seconds}s）")

            if not resp_path.exists():
                err = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
                raise SandboxError(
                    f"子进程未产出结果（退出码 {proc.returncode}）"
                    + (f"：{err.strip()}" if err.strip() else "")
                )
            try:
                return json.loads(resp_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                raise SandboxError(f"解析子进程结果失败：{e}")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def introspect(self, file_path: Path) -> List[dict]:
        """在子进程里导入文件，返回 [{name, description, args}]。失败抛 SandboxError。"""
        data = self._run("introspect", file_path, None)
        if not data.get("ok"):
            raise SandboxError(data.get("error") or "导入失败")
        return data.get("specs", [])

    def invoke(self, file_path: Path, name: str, kwargs: dict) -> str:
        """在子进程里调用某个工具，返回字符串结果；任何失败都以字符串形式返回，不抛。"""
        try:
            data = self._run("invoke", file_path, {"name": name, "args": kwargs or {}})
        except SandboxError as e:
            return f"[沙箱错误] {e}"
        if not data.get("ok"):
            return f"[工具错误] {data.get('error') or '执行失败'}"
        result = data.get("result")
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def build_proxy_tools(self, file_path: Path) -> List[Any]:
        """introspect 后为每个工具造一个转发到子进程的 StructuredTool 代理。"""
        specs = self.introspect(file_path)
        return [self._make_proxy(file_path, spec) for spec in specs]

    # ------------------------------------------------------------------
    # 代理工具
    # ------------------------------------------------------------------
    def _make_proxy(self, file_path: Path, spec: dict) -> Any:
        import asyncio

        from langchain_core.tools import StructuredTool

        name = spec["name"]
        description = spec.get("description") or name
        args_schema = _build_args_model(
            name, spec.get("args") or {}, spec.get("required") or []
        )

        runner = self

        def _clean(kwargs: dict) -> dict:
            # 丢弃值为 None 的可选参数：让子进程里的工具用自身默认值，
            # 避免把重建 schema 的 None 误塞进真实必填类型（如 bool）。
            return {k: v for k, v in kwargs.items() if v is not None}

        def _proxy(**kwargs) -> str:
            return runner.invoke(file_path, name, _clean(kwargs))

        async def _aproxy(**kwargs) -> str:
            return await asyncio.to_thread(runner.invoke, file_path, name, _clean(kwargs))

        return StructuredTool.from_function(
            func=_proxy,
            coroutine=_aproxy,
            name=name,
            description=description,
            args_schema=args_schema,
        )


# JSON schema 类型 → python 类型
_JSON_TYPE = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _build_args_model(tool_name: str, args: Dict[str, dict], required: Optional[List[str]] = None):
    """用 introspect 拿到的参数 schema 重建 pydantic 模型，供代理工具用。"""
    from typing import Any as _Any

    from pydantic import Field, create_model

    required_set = set(required or [])
    fields: Dict[str, Any] = {}
    for pname, pschema in (args or {}).items():
        pschema = pschema if isinstance(pschema, dict) else {}
        py_type = _JSON_TYPE.get(pschema.get("type"), _Any)
        desc = pschema.get("description")
        if pname in required_set:
            fields[pname] = (py_type, Field(..., description=desc) if desc else ...)
        else:
            fields[pname] = (Optional[py_type], Field(None, description=desc) if desc else None)
    if not fields:
        return None
    return create_model(f"{tool_name}_Args", **fields)
