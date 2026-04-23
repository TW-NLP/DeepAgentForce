# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/tianwei/paper/DeepAgentForce/main.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/tianwei/paper/DeepAgentForce/static', 'static'), ('/Users/tianwei/paper/DeepAgentForce/images', 'images'), ('/Users/tianwei/paper/DeepAgentForce/src/services/skills', 'src/services/skills'), ('/Users/tianwei/paper/DeepAgentForce/build/macos/.env', '.env')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeepAgentForce',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['/Users/tianwei/paper/DeepAgentForce/build/macos/app.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepAgentForce',
)
app = BUNDLE(
    coll,
    name='DeepAgentForce.app',
    icon='/Users/tianwei/paper/DeepAgentForce/build/macos/app.icns',
    bundle_identifier=None,
)
