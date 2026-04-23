#!/usr/bin/env python3
"""
Build a desktop distributable for DeepAgentForce with PyInstaller.

This script creates a one-folder app bundle that includes:
- the FastAPI backend entrypoint
- static frontend assets
- image assets
- bundled skill resources

It also writes a runtime .env file for the packaged app so the desktop
bundle can use a local SQLite database without external services.

Usage:
    python packaging/build_desktop.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / "build" / "desktop"
DIST_ROOT = REPO_ROOT / "dist"
APP_NAME = "DeepAgentForce"


def _data_arg(src: Path, dest: str) -> str:
    if os.name == "nt":
        return f"{src};{dest}"
    return f"{src}:{dest}"


def _build_runtime_env() -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    env_path = BUILD_ROOT / ".env"

    lines = [
        "HOST=127.0.0.1",
        "PORT=8000",
        "FRONTEND_HOST=127.0.0.1",
        "FRONTEND_PORT=8000",
        "AUTO_OPEN_BROWSER=false",
        "SQLITE_DB_PATH=deepagentforce.db",
    ]

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def _run_pyinstaller(env_path: Path) -> None:
    try:
        import PyInstaller.__main__  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed. Install it with `pip install pyinstaller` first."
        ) from exc

    try:
        import webview  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pywebview is not installed. Install it with `pip install pywebview` first."
        ) from exc

    _ = webview  # keep the import explicit for hidden-import parity

    static_dir = REPO_ROOT / "static"
    images_dir = REPO_ROOT / "images"
    skills_dir = REPO_ROOT / "src" / "services" / "skills"
    main_py = REPO_ROOT / "main.py"
    extra_hidden_imports = ["webview"]
    if sys.platform == "darwin":
        extra_hidden_imports.extend(["AppKit", "Foundation", "objc"])

    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        APP_NAME,
        "--hidden-import",
        "aiosqlite",
        "--hidden-import",
        "sqlalchemy.dialects.sqlite.aiosqlite",
        "--collect-submodules",
        "webview",
        "--add-data",
        _data_arg(static_dir, "static"),
        "--add-data",
        _data_arg(images_dir, "images"),
        "--add-data",
        _data_arg(skills_dir, "src/services/skills"),
        "--add-data",
        _data_arg(env_path, ".env"),
        str(main_py),
    ]

    for module in extra_hidden_imports:
        args.extend(["--hidden-import", module])

    PyInstaller.__main__.run(args)


def _copy_runtime_env_into_dist(env_path: Path) -> None:
    app_dist = DIST_ROOT / APP_NAME
    app_dist.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_path, app_dist / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DeepAgentForce desktop bundle")
    parser.parse_args()

    env_path = _build_runtime_env()
    _run_pyinstaller(env_path)
    _copy_runtime_env_into_dist(env_path)
    print(f"Build complete. App bundle is under: {DIST_ROOT / APP_NAME}")
    print(f"Runtime env file written to: {DIST_ROOT / APP_NAME / '.env'}")


if __name__ == "__main__":
    main()
