#!/usr/bin/env python
"""验证自定义工具子进程沙箱：隔离、超时、密钥过滤、代理工具调用。

    PY=/Users/tianwei/miniforge3/envs/agent/bin/python
    $PY scripts/test_sandbox.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    ok = bool(ok)
    PASS += ok
    FAIL += (not ok)
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  —— {detail}" if detail else ""))


def write(d: Path, name: str, code: str) -> Path:
    p = d / name
    p.write_text(code, encoding="utf-8")
    return p


def main() -> int:
    from src.services.sandbox import SandboxRunner, SandboxError

    runner = SandboxRunner(wall_seconds=8)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # 1) 基本 introspect + invoke
        f_ok = write(d, "ok.py", (
            'def square(n: int) -> int:\n'
            '    """返回 n 的平方。"""\n'
            '    return n * n\n\n'
            'def greet(name: str, excited: bool = False) -> str:\n'
            '    """打招呼。"""\n'
            '    return ("HELLO " if excited else "hi ") + name\n'
        ))
        specs = runner.introspect(f_ok)
        names = sorted(s["name"] for s in specs)
        check("introspect 抽到两个工具", names == ["greet", "square"], str(names))
        sq = next(s for s in specs if s["name"] == "square")
        check("square 必填参数 n", sq.get("required") == ["n"], str(sq.get("required")))
        gr = next(s for s in specs if s["name"] == "greet")
        check("greet 必填仅 name（excited 可选）", gr.get("required") == ["name"], str(gr.get("required")))

        check("invoke square(6)=36", runner.invoke(f_ok, "square", {"n": 6}) == "36")
        check("invoke greet", runner.invoke(f_ok, "greet", {"name": "Tom", "excited": True}) == "HELLO Tom")
        check("invoke 不存在的工具返回错误串",
              "不存在" in runner.invoke(f_ok, "nope", {}))

        # 2) 代理工具：func + 异步均转发到子进程
        proxies = {t.name: t for t in runner.build_proxy_tools(f_ok)}
        check("代理工具数量=2", len(proxies) == 2)
        check("代理 square.invoke", proxies["square"].invoke({"n": 8}) == "64")
        check("代理工具描述保留", proxies["square"].description == "返回 n 的平方。")
        # 必填参数缺失 → schema 校验前置（不进沙箱也会报）
        import asyncio
        aval = asyncio.run(proxies["greet"].ainvoke({"name": "Ann"}))
        check("代理异步 ainvoke", aval == "hi Ann", str(aval))

        # 3) 导入期异常 → SandboxError（不污染主进程）
        f_boom = write(d, "boom.py", 'raise RuntimeError("import boom")\n')
        try:
            runner.introspect(f_boom)
            check("导入期异常抛 SandboxError", False, "未抛")
        except SandboxError as e:
            check("导入期异常抛 SandboxError", "boom" in str(e) or "RuntimeError" in str(e), str(e))

        # 4) 运行期异常 → invoke 返回错误串，不抛
        f_runerr = write(d, "runerr.py", (
            'def boom() -> str:\n'
            '    """运行期炸。"""\n'
            '    raise ValueError("kaboom")\n'
        ))
        res = runner.invoke(f_runerr, "boom", {})
        check("运行期异常以错误串返回", "kaboom" in res or "ValueError" in res, res)

        # 5) 墙钟超时
        slow = SandboxRunner(wall_seconds=2)
        f_slow = write(d, "slow.py", (
            'import time\n'
            'def sleeper() -> str:\n'
            '    """睡很久。"""\n'
            '    time.sleep(30)\n'
            '    return "done"\n'
        ))
        res = slow.invoke(f_slow, "sleeper", {})
        check("超时被拦截", "超时" in res or "沙箱错误" in res, res)

        # 6) 密钥环境变量被过滤
        os.environ["MY_SECRET_TOKEN"] = "should-not-leak"
        os.environ["SAFE_VALUE"] = "ok-to-see"
        f_env = write(d, "env.py", (
            'import os\n'
            'def peek() -> str:\n'
            '    """看环境变量。"""\n'
            '    return f"secret={os.environ.get(\'MY_SECRET_TOKEN\',\'<none>\')};safe={os.environ.get(\'SAFE_VALUE\',\'<none>\')}"\n'
        ))
        res = runner.invoke(f_env, "peek", {})
        check("密钥变量被过滤", "secret=<none>" in res, res)
        check("非密钥变量保留", "safe=ok-to-see" in res, res)

    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
