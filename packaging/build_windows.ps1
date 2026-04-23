$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildDir = Join-Path $RootDir "build\windows"
$DistDir = Join-Path $RootDir "dist"
$AppName = "DeepAgentForce"
$AppBundle = Join-Path $DistDir $AppName
$EnvPath = Join-Path $BuildDir ".env"
$IconPath = Join-Path $BuildDir "app.ico"
$IconPreview = Join-Path $BuildDir "app_round.png"
$LogoSrc = Join-Path $RootDir "images\logo.png"

Write-Host "Building $AppName for Windows..." -ForegroundColor Green

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

# Check required Python packages
Write-Host "Checking dependencies..." -ForegroundColor Yellow
python -c "import PIL, PyInstaller, webview"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pillow, PyInstaller, and pywebview must be installed in the active Python environment."
    exit 1
}
Write-Host "✓ All dependencies found" -ForegroundColor Green

# Generate app icon
Write-Host "Generating app icons..." -ForegroundColor Yellow
python (Join-Path $RootDir "packaging\make_app_icon.py") `
    --source $LogoSrc `
    --png $IconPreview `
    --ico $IconPath

# Create .env file
Write-Host "Creating .env configuration..." -ForegroundColor Yellow
$EnvContent = @"
HOST=127.0.0.1
PORT=8000
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=8000
AUTO_OPEN_BROWSER=false
SQLITE_DB_PATH=deepagentforce.db
"@
Set-Content -Path $EnvPath -Value $EnvContent -Encoding UTF8

# Run PyInstaller
Write-Host "Running PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name $AppName `
    --distpath $DistDir `
    --workpath (Join-Path $BuildDir "pyinstaller-work") `
    --specpath $BuildDir `
    --icon $IconPath `
    --hidden-import aiosqlite `
    --hidden-import sqlalchemy.dialects.sqlite.aiosqlite `
    --hidden-import webview `
    --collect-submodules webview `
    --add-data "$RootDir\static;static" `
    --add-data "$RootDir\images;images" `
    --add-data "$RootDir\src\services\skills;src/services/skills" `
    --add-data "$EnvPath;." `
    (Join-Path $RootDir "main.py")

# Copy .env to the built app
New-Item -ItemType Directory -Force -Path $AppBundle | Out-Null
Copy-Item $EnvPath (Join-Path $AppBundle ".env") -Force

Write-Host "✓ Build complete!" -ForegroundColor Green
Write-Host "Application bundle: $AppBundle" -ForegroundColor Green

$InnoScript = Join-Path $BuildDir "installer.iss"
@"
[Setup]
AppName=$AppName
AppVersion=1.0.0
DefaultDirName={autopf}\$AppName
DefaultGroupName=$AppName
OutputDir=$DistDir
OutputBaseFilename=$AppName-Setup
Compression=lzma
SolidCompression=yes
SetupIconFile="$IconPath"

[Files]
Source: "$AppBundle\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\$AppName"; Filename: "{app}\$AppName.exe"
Name: "{commondesktop}\$AppName"; Filename: "{app}\$AppName.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
"@ | Set-Content -Encoding UTF8 $InnoScript

if (Get-Command ISCC.exe -ErrorAction SilentlyContinue) {
    & ISCC.exe $InnoScript
    Write-Host "Created Windows installer in $DistDir"
} else {
    $ZipPath = Join-Path $DistDir "$AppName-windows.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path (Join-Path $AppBundle "*") -DestinationPath $ZipPath -Force
    Write-Host "ISCC not found. Created zip bundle instead: $ZipPath"
}
