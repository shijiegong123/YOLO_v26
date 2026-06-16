import torch
import ultralytics
import platform
import os


# ============================================================
# 内置模型列表定义
# ============================================================
# 格式: { '显示名': {'file': '文件名', 'task': '任务类型', 'desc': '描述'} }

BUILTIN_MODELS = {
    # ---- 检测模型 (Detect) ----
    "YOLO26-Nano (检测)":     {"file": "yolo26n.pt",     "task": "detect",  "desc": "最快速，适合实时检测"},
    "YOLO26-Small (检测)":    {"file": "yolo26s.pt",     "task": "detect",  "desc": "速度与精度平衡"},
    "YOLO26-Medium (检测)":   {"file": "yolo26m.pt",     "task": "detect",  "desc": "更高精度"},
    "YOLO26-Large (检测)":    {"file": "yolo26l.pt",     "task": "detect",  "desc": "高精度，速度较慢"},
    "YOLO26-XLarge (检测)":   {"file": "yolo26x.pt",     "task": "detect",  "desc": "最高精度"},
    # ---- 分割模型 (Segment) ----
    "YOLO26-Nano (分割)":     {"file": "yolo26n-seg.pt", "task": "segment", "desc": "最快速实例分割"},
    "YOLO26-Small (分割)":    {"file": "yolo26s-seg.pt", "task": "segment", "desc": "速度与精度平衡分割"},
    "YOLO26-Medium (分割)":   {"file": "yolo26m-seg.pt", "task": "segment", "desc": "更高精度分割"},
    # ---- 姿态估计模型 (Pose) ----
    "YOLO26-Nano (姿态)":     {"file": "yolo26n-pose.pt","task": "pose",    "desc": "最快速姿态估计"},
    "YOLO26-Small (姿态)":    {"file": "yolo26s-pose.pt","task": "pose",    "desc": "速度与精度平衡姿态"},
    "YOLO26-Medium (姿态)":   {"file": "yolo26m-pose.pt","task": "pose",    "desc": "更高精度姿态估计"},
    # ---- OBB 旋转框模型 ----
    "YOLO26-Nano (OBB)":      {"file": "yolo26n-obb.pt", "task": "obb",     "desc": "最快速旋转框检测"},
    "YOLO26-Small (OBB)":     {"file": "yolo26s-obb.pt", "task": "obb",     "desc": "速度与精度平衡旋转框"},
    # ---- 经典兼容模型 ----
    "YOLO11-Nano (检测)":     {"file": "yolo11n.pt",     "task": "detect",  "desc": "YOLO11 经典模型"},
}


def get_builtin_models():
    """
    返回内置模型列表，供 GUI 下拉框使用。
    返回格式: list of (显示名, 文件路径, 任务类型, 描述)
    """
    return [
        (name, info["file"], info["task"], info["desc"])
        for name, info in BUILTIN_MODELS.items()
    ]


def get_model_task(model_file: str) -> str:
    """
    根据模型文件名推断其任务类型。
    :param model_file: 模型文件名 (如 yolo26s-seg.pt)
    :return: 任务类型字符串
    """
    name = model_file.lower()
    if "-seg" in name:
        return "segment"
    elif "-pose" in name:
        return "pose"
    elif "-obb" in name:
        return "obb"
    elif "-cls" in name:
        return "classify"
    return "detect"


def format_size(size_bytes: float) -> str:
    """
    将字节数格式化为可读文件大小字符串。
    :param size_bytes: 字节数
    :return: 如 '12.3 MB'
    """
    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def model_exists_locally(model_file: str) -> bool:
    """
    检查模型文件是否存在于本地（当前目录或 Ultralytics 默认缓存目录）。
    :param model_file: 模型文件名
    :return: 是否存在
    """
    if os.path.exists(model_file):
        return True
    # Ultralytics 默认下载目录
    cache_dir = os.path.expanduser("~/.config/Ultralytics/settings.yaml")
    if os.path.exists(os.path.join(os.path.dirname(cache_dir), model_file)):
        return True
    return False


def check_environment():
    """
    检查 CUDA 可用性并打印环境信息。
    :return: 设备名称 ('cuda' 或 'cpu')
    """
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'

    print("=" * 60)
    print("环境检测信息:")
    print(f"  操作系统      : {platform.system()} {platform.release()}")
    print(f"  Python 版本   : {platform.python_version()}")
    print(f"  Ultralytics   : {ultralytics.__version__}")
    print(f"  PyTorch 版本  : {torch.__version__}")
    print(f"  CUDA 可用     : {cuda_available}")

    if cuda_available:
        print(f"  CUDA 设备     : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本     : {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU 内存      : {total_mem:.1f} GB")

    print(f"  使用设备      : {device}")
    print("=" * 60)

    return device


class BaseYOLOLoader:
    """YOLO 模型加载器基类"""

    def __init__(self):
        self.device = check_environment()
        self.model = None
        self.current_model_path = None

    def is_loaded(self):
        """检查模型是否已加载"""
        return self.model is not None

    def unload_model(self):
        """卸载模型，释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_path = None
            # 清理 CUDA 缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            print("模型已卸载，GPU 缓存已释放")
