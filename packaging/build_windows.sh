#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/windows"
DIST_DIR="$ROOT_DIR/dist"
APP_NAME="DeepAgentForce"
APP_BUNDLE="$DIST_DIR/$APP_NAME"
ENV_PATH="$BUILD_DIR/.env"
ICON_PATH="$BUILD_DIR/app.ico"
ICON_PREVIEW="$BUILD_DIR/app_round.png"
LOGO_SRC="$ROOT_DIR/images/logo.png"

mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

echo "Building $APP_NAME for Windows..."

# Check required Python packages
python - <<'PY'
import importlib
import sys

required = ["PIL", "PyInstaller", "webview"]
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

echo "✓ All dependencies found"

# Generate app icon
echo "Generating app icons..."
python "$ROOT_DIR/packaging/make_app_icon.py" \
    --source "$LOGO_SRC" \
    --png "$ICON_PREVIEW" \
    --ico "$ICON_PATH"

# Create .env file
echo "Creating .env configuration..."
cat > "$ENV_PATH" <<EOF
HOST=127.0.0.1
PORT=8000
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=8000
AUTO_OPEN_BROWSER=false
SQLITE_DB_PATH=deepagentforce.db
EOF

# PyInstaller - use colon separator for macOS/Linux, semicolon for Windows
echo "Running PyInstaller..."
SEPARATOR=":"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    SEPARATOR=";"
fi

python -m PyInstaller \
    --noconfirm \
    --clean \
    --onedir \
    --name "$APP_NAME" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR/pyinstaller-work" \
    --specpath "$BUILD_DIR" \
    --icon "$ICON_PATH" \
    --hidden-import aiosqlite \
    --hidden-import sqlalchemy.dialects.sqlite.aiosqlite \
    --hidden-import webview \
    --collect-submodules webview \
    --add-data "$ROOT_DIR/static${SEPARATOR}static" \
    --add-data "$ROOT_DIR/images${SEPARATOR}images" \
    --add-data "$ROOT_DIR/src/services/skills${SEPARATOR}src/services/skills" \
    --add-data "$ENV_PATH${SEPARATOR}." \
    "$ROOT_DIR/main.py"

# Copy .env to the built app
mkdir -p "$APP_BUNDLE"
cp "$ENV_PATH" "$APP_BUNDLE/.env"

echo "✓ Build complete!"
echo "Application bundle: $APP_BUNDLE"
