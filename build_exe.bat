@echo off
chcp 437 > nul
title ThyroidAnalysis - Build EXE

echo ============================================================
echo   Thyroid Ultrasound Analysis System - PyInstaller Build
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [2/4] Cleaning old build files...
if exist build  rmdir /s /q build
if exist dist   rmdir /s /q dist
if exist ThyroidAnalysis.spec del /f ThyroidAnalysis.spec
echo       Done.
echo.

echo [3/4] Building EXE with PyInstaller...
echo       Note: PyTorch is large. Final EXE ~1-2 GB.
echo.

pyinstaller ^
    --onefile ^
    --windowed ^
    --name ThyroidAnalysis ^
    --add-data "models;models" ^
    --add-data "utils;utils" ^
    --add-data "config.py;." ^
    --add-data "inference.py;." ^
    --hidden-import=torch ^
    --hidden-import=torch.nn ^
    --hidden-import=torchvision ^
    --hidden-import=torchvision.models ^
    --hidden-import=cv2 ^
    --hidden-import=PIL ^
    --hidden-import=numpy ^
    --collect-all torch ^
    --collect-all torchvision ^
    --collect-all cv2 ^
    --noconfirm ^
    gui_app.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed!
    echo Common causes:
    echo   1. Incomplete torch/torchvision installation
    echo   2. Insufficient disk space (5 GB+ required)
    echo   3. Antivirus blocking - try disabling temporarily
    pause
    exit /b 1
)

echo.
echo [4/4] Build successful!
echo.
echo ============================================================
echo   Output: dist\ThyroidAnalysis.exe
echo ============================================================
echo.
echo IMPORTANT:
echo   Place checkpoints\ folder next to ThyroidAnalysis.exe
echo   Or just run:  python gui_app.py  (no packaging needed)
echo.
pause