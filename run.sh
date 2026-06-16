#!/bin/bash
# YOLO 智能检测平台启动脚本

echo "======================================"
echo "  Ultralytics YOLO 智能检测平台"
echo "======================================"
echo ""

# 检查 Python 是否安装
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "Python 版本: $(python --version)"
echo ""

# 检查依赖
echo "检查依赖..."
python verify_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "正在启动程序..."
    echo ""
    python app_main.py
else
    echo ""
    echo "依赖检查失败，请安装依赖:"
    echo "pip install -r requirements.txt"
    exit 1
fi
