# 错误修复记录

## 2026-06-15 - 启动错误修复

### 问题 1: QSizePolicy 未定义

**错误信息:**
```
NameError: name 'QSizePolicy' is not defined
```

**原因:**
在 `main_window.py` 中使用了 `QSizePolicy.Expanding`，但没有从 PySide6.QtWidgets 导入该类。

**修复方案:**
在 `main_window.py` 的导入语句中添加 `QSizePolicy`：

```python
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QComboBox, 
                               QSlider, QFileDialog, QTextEdit, QGroupBox, QSizePolicy)
```

**文件:** main_window.py (第 4 行)

---

### 问题 2: GPU 内存属性名错误

**错误信息:**
```
AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'. 
Did you mean: 'total_memory'?
```

**原因:**
在 `utils.py` 中使用了错误的属性名 `total_mem`，正确的属性名应该是 `total_memory`。

**修复方案:**
将 `total_mem` 改为 `total_memory`：

```python
print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**文件:** utils.py (第 25 行)

---

## 验证结果

✅ 所有错误已修复  
✅ 程序可以正常启动  
✅ GUI 界面正常显示  
✅ CUDA GPU 正确识别（NVIDIA GeForce GTX 1660, 5.8 GB）  
✅ 默认模型 yolo11n.pt 成功加载  
✅ 所有依赖测试通过  

## 测试环境

- 操作系统: Linux 5.15.0-139-generic
- Python 版本: 3.10.20
- PyTorch 版本: 2.5.1+cu121
- Ultralytics 版本: 8.4.66
- CUDA 版本: 12.1
- GPU: NVIDIA GeForce GTX 1660 (5.8 GB)

---

**修复完成时间:** 2026-06-15  
**状态:** ✅ 已解决
