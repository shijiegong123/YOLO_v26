@echo off
chcp 65001 >nul
echo ======================================
echo   Ultralytics YOLO 智能检测平台
echo ======================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo Python 版本:
python --version
echo.

REM 检查依赖
echo 检查依赖...
python verify_installation.py

if errorlevel 1 (
    echo.
    echo 依赖检查失败，请安装依赖:
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 正在启动程序...
echo.
python app_main.py

pause
