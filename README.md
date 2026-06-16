# Ultralytics YOLO 智能检测平台

基于 Ultralytics YOLO 和 PySide6 的桌面端目标检测应用，支持实时视频检测和图像识别。

## 功能特性

- ✅ 支持多种 YOLO 任务类型：检测 (Detect)、分割 (Segment)、姿态估计 (Pose)、旋转框 (OBB)
- ✅ 实时摄像头检测
- ✅ 视频文件检测
- ✅ 图片检测
- ✅ 模型自动下载（首次使用时）
- ✅ GPU/CPU 自动切换
- ✅ 实时 FPS 和目标数量显示
- ✅ 可调节置信度阈值
- ✅ 多线程处理，界面流畅

## 系统要求

- Python >= 3.8
- CUDA (可选，用于 GPU 加速)
- 操作系统: Windows/Linux/macOS

## 安装步骤

### 1. 安装依赖

```bash
cd /home/xszn/PyTorchPro/yolo
pip install -r requirements.txt
```

如果使用 conda 环境：

```bash
conda activate pytorch
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python app_main.py
```

## 使用说明

### 加载模型

1. 点击 "加载模型 (.pt)" 按钮
2. 选择本地的 `.pt` 模型文件
3. 如果不选择文件直接确认，将自动下载默认的 `yolo11n.pt` 模型

**支持的模型：**
- yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
- yolo11-seg.pt (分割)
- yolo11-pose.pt (姿态估计)
- 自定义训练的 YOLO 模型

### 选择任务类型

在顶部下拉菜单中选择：
- **detect**: 目标检测（默认）
- **segment**: 实例分割
- **pose**: 人体姿态估计
- **obb**: 旋转目标检测

### 调整置信度

使用滑块调整置信度阈值（0.10 - 1.00），默认 0.25
- 较低值：检测到更多目标，但可能有误检
- 较高值：检测结果更准确，但可能漏检

### 开始检测

#### 摄像头检测
1. 确保摄像头已连接
2. 在 "视频源" 下拉菜单选择 "摄像头 (0)"
3. 点击 "开始检测"

#### 视频文件检测
1. 在 "视频源" 下拉菜单选择 "视频文件..."
2. 选择要检测的视频文件（支持 .mp4, .avi, .mov, .mkv）
3. 点击 "开始检测"

### 停止检测

点击 "停止检测" 按钮即可停止当前检测任务。

## 项目结构

```
yolo/
├── app_main.py          # 主入口文件和控制器
├── main_window.py       # GUI 界面设计
├── yolo_detector.py     # YOLO 检测器核心逻辑
├── worker_thread.py     # 多线程处理（防止界面卡顿）
├── utils.py             # 工具函数和环境检测
├── requirements.txt     # Python 依赖列表
├── yolo_app.spec        # PyInstaller 打包配置
└── README.md            # 项目说明文档
```

## 高级功能

### GPU 加速

如果系统有 NVIDIA GPU 并安装了 CUDA，程序会自动使用 GPU 进行推理加速。

检查 GPU 是否可用：
```python
import torch
print(torch.cuda.is_available())
```

### 性能优化建议

1. **选择合适的模型大小**：
   - nano (n): 最快，精度较低
   - small (s): 平衡速度和精度
   - medium (m): 较慢，精度高
   - large (l), xlarge (x): 最慢，精度最高

2. **调整图像尺寸**：
   - 代码中默认为 640x640
   - 可以在 `worker_thread.py` 中修改 `imgsz` 参数

3. **使用半精度推理**：
   - GPU 模式下自动启用 FP16
   - 可显著提升推理速度

## 打包发布

使用 PyInstaller 打包为独立可执行文件：

```bash
pyinstaller yolo_app.spec
```

生成的可执行文件位于 `dist/` 目录中。

## 常见问题

### Q: 提示 "No module named 'cv2'"
A: 安装 opencv-python：
```bash
pip install opencv-python
```

### Q: Linux 下无法显示窗口
A: 安装必要的系统依赖：
```bash
sudo apt-get install libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxcb-xkb1 \
    libxkbcommon-x11-0 libxkbcommon0
```

### Q: 模型下载很慢
A: 可以手动下载模型文件放到项目目录，或配置代理。

### Q: GPU 内存不足
A: 
1. 使用更小的模型（如 yolo11n）
2. 减小 imgsz 参数（如改为 416）
3. 关闭其他占用 GPU 的程序

## 技术栈

- **深度学习框架**: PyTorch + Ultralytics YOLO
- **GUI 框架**: PySide6
- **图像处理**: OpenCV
- **数值计算**: NumPy
- **打包工具**: PyInstaller

## 许可证

本项目遵循 LGPL 协议（PySide6），商业使用请注意合规性。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**注意**: 本项目使用的 "YOLOv26" 是指 Ultralytics 最新版本的 YOLO 模型系列。实际模型名称可能为 yolo11、yolov8 等，具体取决于安装的 ultralytics 库版本。
