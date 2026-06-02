#!/usr/bin/env python
"""HTTP 层验证 /api/tools 与 /api/mcp 路由（FastAPI TestClient + 真实 JWT）。

    PY=/Users/tianwei/miniforge3/envs/agent/bin/python
    $PY scripts/test_routes_http.py
"""
from __future__ import annotations

import sys
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


def main() -> int:
    import main as app_main
    from src.services.auth_service import auth_service
    from fastapi.testclient import TestClient

    tenant = "http-test-tenant"
    token = auth_service.create_access_token(user_id=1, username="tester", tenant_uuid=tenant)
    H = {"Authorization": f"Bearer {token}"}
    client = TestClient(app_main.app)

    print("=" * 70 + "\n工具路由 /api/tools\n" + "=" * 70)

    r = client.get("/api/tools", headers=H)
    check("GET /tools 200", r.status_code == 200, str(r.status_code))
    body = r.json()
    check("返回内置工具（>=10）", len(body.get("builtin", [])) >= 10, f"{len(body.get('builtin', []))}")
    check("初始无自定义工具", body.get("custom", []) == [])

    r = client.get("/api/tools/template", headers=H)
    check("GET /tools/template", r.status_code == 200 and "def " in r.json().get("code", ""))

    code = ('def square(n: int) -> int:\n'
            '    """返回 n 的平方。"""\n'
            '    return n * n\n')
    r = client.post("/api/tools/validate", headers=H, data={"code": code})
    check("POST /tools/validate 合法", r.json().get("success") and "square" in r.json().get("tool_names", []))

    r = client.post("/api/tools/custom", headers=H,
                    data={"tool_id": "math_pack", "code": code, "force": "false"})
    check("POST /tools/custom 保存", r.json().get("success"), str(r.json()))

    r = client.get("/api/tools", headers=H)
    check("自定义工具已出现", any(c["tool_id"] == "math_pack" for c in r.json().get("custom", [])))

    r = client.get("/api/tools/custom/math_pack", headers=H)
    check("GET 自定义源码", r.status_code == 200 and "square" in r.json().get("code", ""))

    r = client.delete("/api/tools/custom/math_pack", headers=H)
    check("DELETE 自定义工具", r.json().get("success"))
    r = client.get("/api/tools", headers=H)
    check("删除后消失", not any(c["tool_id"] == "math_pack" for c in r.json().get("custom", [])))

    # 未登录访问
    r = client.post("/api/tools/custom", data={"tool_id": "x", "code": code})
    check("未登录保存被拒(401)", r.status_code == 401, str(r.status_code))

    print("\n" + "=" * 70 + "\nMCP 路由 /api/mcp\n" + "=" * 70)

    r = client.get("/api/mcp/servers", headers=H)
    check("GET /mcp/servers 200", r.status_code == 200, str(r.status_code))

    r = client.post("/api/mcp/servers", headers=H, json={
        "name": "demo_http", "transport": "streamable_http", "url": "http://localhost:9/mcp",
    })
    check("POST 新增 server", r.json().get("success"), str(r.json()))

    r = client.get("/api/mcp/servers", headers=H)
    found = next((s for s in r.json().get("servers", []) if s["name"] == "demo_http"), None)
    check("新增 server 可见且可编辑", found and found["editable"] and found["enabled"], str(found))

    r = client.post("/api/mcp/servers/demo_http/toggle", headers=H, json={"enabled": False})
    check("toggle 禁用", r.json().get("success"))
    r = client.get("/api/mcp/servers", headers=H)
    found = next((s for s in r.json().get("servers", []) if s["name"] == "demo_http"), None)
    check("禁用状态生效", found and found["enabled"] is False)

    r = client.post("/api/mcp/servers", headers=H, json={
        "name": "bad name!", "transport": "stdio", "command": "echo",
    })
    check("非法名称被拒", not r.json().get("success"))

    r = client.delete("/api/mcp/servers/demo_http", headers=H)
    check("DELETE server", r.json().get("success"))

    r = client.delete("/api/mcp/servers/nope", headers=H)
    check("删除不存在的返回 404", r.status_code == 404, str(r.status_code))

    # 清理租户文件
    try:
        (Path(app_main.engine.settings.DATA_DIR) / f"mcp_servers_{tenant}.json").unlink(missing_ok=True)
        import shutil
        shutil.rmtree(Path(app_main.engine.settings.DATA_DIR) / "agent_tools_custom" / tenant,
                      ignore_errors=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
