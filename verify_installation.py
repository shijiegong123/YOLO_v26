#!/usr/bin/env python3
"""
YOLO 项目依赖测试脚本
用于验证所有必需的库是否正确安装
"""

def test_imports():
    """测试所有必需的导入"""
    print("="*60)
    print("开始测试项目依赖...")
    print("="*60)
    
    tests = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Ultralytics YOLO", "ultralytics"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("PySide6", "PySide6"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_name in tests:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            print(f"✅ {name:20s} - 版本: {version}")
            passed += 1
        except ImportError as e:
            print(f"❌ {name:20s} - 错误: {str(e)}")
            failed += 1
    
    print("="*60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*60)
    
    if failed > 0:
        print("\n请运行以下命令安装缺失的依赖:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖已正确安装！")
        
        # 额外检查 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA 可用 - 设备: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA 不可用，将使用 CPU 模式")
        except:
            pass
        
        return True


def test_yolo_model():
    """测试 YOLO 模型加载"""
    print("\n" + "="*60)
    print("测试 YOLO 模型加载...")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import os
        
        # 检查是否有本地模型文件
        model_path = "yolo11n.pt"
        if os.path.exists(model_path):
            print(f"找到本地模型: {model_path}")
            print("正在加载模型...")
            model = YOLO(model_path)
            print("✅ 模型加载成功！")
            
            # 显示模型信息
            print(f"   模型任务: {model.task}")
            print(f"   类别数量: {len(model.names)}")
        else:
            print(f"未找到本地模型文件: {model_path}")
            print("首次运行时会自动下载模型")
        
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


if __name__ == "__main__":
    # 测试导入
    imports_ok = test_imports()
    
    if imports_ok:
        # 测试模型
        test_yolo_model()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
