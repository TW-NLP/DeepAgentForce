"""沙箱子进程入口（CLI）。由 :class:`SandboxRunner` 以独立进程拉起。

用法::

    python tool_worker.py <mode> <file_path> <request_json> <response_json>

mode:
- ``introspect``：导入 <file_path>，把每个工具的 {name, description, args} 写入
  <response_json>。
- ``invoke``：从 <request_json> 读 {name, args}，调用对应工具，把
  {ok, result} 或 {ok:false, error} 写入 <response_json>。

**为何用文件而非 stdout 通信**：用户代码里的 ``print()`` 会污染 stdout，故结果统一
落到 <response_json>；stdout/stderr 任其自由。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _project_root() -> Path:
    # tool_worker.py -> sandbox -> services -> src -> <root>
    return Path(__file__).resolve().parents[3]


def _write(response_path: str, payload: dict) -> None:
    Path(response_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main(argv: list) -> int:
    if len(argv) < 5:
        return 2
    mode, file_path, request_path, response_path = argv[1], argv[2], argv[3], argv[4]

    # 让子进程能 import 项目内的 src.services.sandbox.loader
    sys.path.insert(0, str(_project_root()))

    try:
        from src.services.sandbox.loader import load_tools_from_file, tool_specs

        if mode == "introspect":
            specs = tool_specs(Path(file_path))
            _write(response_path, {"ok": True, "specs": specs})
            return 0

        if mode == "invoke":
            req = json.loads(Path(request_path).read_text(encoding="utf-8"))
            name = req.get("name")
            args = req.get("args") or {}
            tools = load_tools_from_file(Path(file_path))
            target = next((t for t in tools if t.name == name), None)
            if target is None:
                _write(response_path, {"ok": False, "error": f"工具 '{name}' 不存在"})
                return 0
            result = target.invoke(args)
            _write(response_path, {"ok": True, "result": _to_jsonable(result)})
            return 0

        _write(response_path, {"ok": False, "error": f"未知 mode: {mode}"})
        return 2
    except Exception as e:  # noqa: BLE001
        try:
            _write(
                response_path,
                {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()},
            )
        except Exception:
            pass
        return 1


def _to_jsonable(value):
    """尽量把工具返回值转成可 JSON 序列化的形式。"""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:  # noqa: BLE001
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
