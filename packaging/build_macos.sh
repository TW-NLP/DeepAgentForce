#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_NAME="DeepAgentForce"
BUILD_DIR="$ROOT_DIR/build/macos"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
DMG_NAME="$DIST_DIR/$APP_NAME-macOS.dmg"
ICNS_FILE="$BUILD_DIR/app.icns"
BASE_ICON_PNG="$BUILD_DIR/app_logo_round.png"
ICONSET_DIR="$BUILD_DIR/app.iconset"
LOGO_SRC="$ROOT_DIR/images/logo.png"

mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"
mkdir -p "$BUILD_DIR/pyinstaller-config"
mkdir -p "$BUILD_DIR/pyinstaller-work"

export PYINSTALLER_CONFIG_DIR="$BUILD_DIR/pyinstaller-config"

PYTHON_BIN="$(python - <<'PY'
import sys
print(sys.executable)
PY
)"
echo "Using Python interpreter: $PYTHON_BIN"

python - <<'PY'
import importlib
import importlib.util
import sys

required = ["deepagents", "aiosqlite", "PyInstaller", "PIL", "webview", "AppKit", "objc", "Foundation"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print(f"Current Python interpreter: {sys.executable}", file=sys.stderr)
    print("Missing build dependencies in the current Python environment:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)
PY

# PyInstaller 与第三方 pathlib backport 不兼容；如果环境里被拉回来了，先清理掉。
python -m pip uninstall -y pathlib >/dev/null 2>&1 || true

# 生成 macOS 图标源图（圆角正方形 PNG）和 .icns
python "$ROOT_DIR/packaging/make_app_icon.py" \
  --source "$LOGO_SRC" \
  --png "$BASE_ICON_PNG" \
  --iconset "$ICONSET_DIR" \
  --icns "$ICNS_FILE"

cat > "$BUILD_DIR/.env" <<EOF
HOST=127.0.0.1
PORT=8000
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=8000
AUTO_OPEN_BROWSER=false
SQLITE_DB_PATH=deepagentforce.db
EOF

python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR/pyinstaller-work" \
  --specpath "$BUILD_DIR" \
  --icon "$ICNS_FILE" \
  --add-data "$ROOT_DIR/static:static" \
  --add-data "$ROOT_DIR/images:images" \
  --add-data "$ROOT_DIR/src/services/skills:src/services/skills" \
  --add-data "$BUILD_DIR/.env:.env" \
  --hidden-import aiosqlite \
  --hidden-import sqlalchemy.dialects.sqlite.aiosqlite \
  --hidden-import webview \
  --hidden-import AppKit \
  --hidden-import Foundation \
  --hidden-import objc \
  --collect-submodules webview \
  "$ROOT_DIR/main.py"

mkdir -p "$APP_BUNDLE/Contents/MacOS"
cp "$BUILD_DIR/.env" "$APP_BUNDLE/Contents/MacOS/.env"

if command -v create-dmg >/dev/null 2>&1; then
  create-dmg \
    --volname "$APP_NAME" \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 96 \
    --app-drop-link 600 185 \
    "$DMG_NAME" \
    "$APP_BUNDLE"
  echo "Created DMG: $DMG_NAME"
else
  hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$APP_BUNDLE" \
    -ov \
    -format UDZO \
    "$DMG_NAME"
  echo "Created DMG: $DMG_NAME"
fi
