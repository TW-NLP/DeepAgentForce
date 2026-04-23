# Desktop Packaging

This project can be bundled as a desktop app with PyInstaller.

## What gets packaged

- FastAPI backend
- Static frontend
- Images and bundled skills
- Runtime `.env` file placed next to the executable / app bundle
- Rounded-square app icon generated from `images/logo.png`
- Packaged desktop app opens in a native window via `pywebview`
- Desktop app data and SQLite files are stored in the user's writable application data directory

## macOS

```bash
chmod +x packaging/build_macos.sh
./packaging/build_macos.sh
```

Outputs:

- `dist/DeepAgentForce/`
- `dist/DeepAgentForce-macOS.dmg`

## Windows

```powershell
.\packaging\build_windows.ps1
```

Outputs:

- `dist\DeepAgentForce\`
- `dist\DeepAgentForce-Setup.exe` if Inno Setup is installed
- `dist\DeepAgentForce-windows.zip` as a fallback

## One-folder build

If you only want the bundle without an installer:

```bash
python packaging/build_desktop.py
```

## Important

- The packaged desktop app uses a local SQLite database by default.
- The bundled backend reads its runtime `.env` from the executable / app bundle directory.
- The default SQLite file path is `data/deepagentforce.db`.
