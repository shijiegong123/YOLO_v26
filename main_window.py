import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QTextEdit, QGroupBox, QSizePolicy, QSplitter,
    QFrame, QToolButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from utils import get_builtin_models


# -----------------------------------------------------------------------
#  全局 QSS 样式表 —— 深色现代主题
# -----------------------------------------------------------------------
DARK_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}
QPushButton#btnStart {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}
QPushButton#btnStart:hover { background-color: #94e2d5; }
QPushButton#btnStop {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}
QPushButton#btnStop:hover { background-color: #eba0ac; }
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
    color: #cdd6f4;
}
QComboBox:hover { border-color: #89b4fa; }
QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
    color: #cdd6f4;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #b4befe;
}
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}
QLabel#lblDisplay {
    background-color: #11111b;
    border: 2px solid #313244;
    border-radius: 8px;
    min-width: 640px;
    min-height: 480px;
}
QLabel#lblModelPath {
    color: #a6adc8;
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


class MainWindow(QMainWindow):
    """YOLO 智能检测平台主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultralytics YOLO 智能检测平台")
        self.setGeometry(80, 60, 1360, 900)
        self.setStyleSheet(DARK_STYLE)
        self._builtin_models = get_builtin_models()
        self.init_ui()

    # ------------------------------------------------------------------
    #  UI 构建
    # ------------------------------------------------------------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # ===================== 顶部工具栏 =====================
        top_group = QGroupBox("模型与参数控制")
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(10)

        # 加载自定义模型
        self.btn_load_model = QPushButton("加载模型 (.pt)")
        self.btn_load_model.setMinimumWidth(120)

        # 内置模型下拉框
        self.cmb_builtin = QComboBox()
        self.cmb_builtin.addItem("-- 选择内置模型 --")
        for name, _file, _task, desc in self._builtin_models:
            self.cmb_builtin.addItem(f"{name}  [{desc}]")
        self.cmb_builtin.setMinimumWidth(260)

        # 模型路径标签
        self.lbl_model_path = QLabel("未选择模型")
        self.lbl_model_path.setObjectName("lblModelPath")
        self.lbl_model_path.setMinimumWidth(160)
        self.lbl_model_path.setMaximumWidth(220)

        # 任务类型
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["detect", "segment", "pose", "obb"])
        self.cmb_task.setMinimumWidth(90)

        # 置信度滑块
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(5, 95)
        self.sld_conf.setValue(25)
        self.sld_conf.setMinimumWidth(120)
        self.lbl_conf_val = QLabel("Conf: 0.25")
        self.lbl_conf_val.setMinimumWidth(75)

        # IOU 滑块
        self.sld_iou = QSlider(Qt.Horizontal)
        self.sld_iou.setRange(10, 95)
        self.sld_iou.setValue(45)
        self.sld_iou.setMinimumWidth(120)
        self.lbl_iou_val = QLabel("IOU: 0.45")
        self.lbl_iou_val.setMinimumWidth(75)

        top_layout.addWidget(self.btn_load_model)
        top_layout.addWidget(QLabel("内置:"))
        top_layout.addWidget(self.cmb_builtin)
        top_layout.addWidget(self.lbl_model_path)
        top_layout.addStretch()
        top_layout.addWidget(QLabel("任务:"))
        top_layout.addWidget(self.cmb_task)
        top_layout.addWidget(QLabel("置信度:"))
        top_layout.addWidget(self.sld_conf)
        top_layout.addWidget(self.lbl_conf_val)
        top_layout.addWidget(QLabel("IOU:"))
        top_layout.addWidget(self.sld_iou)
        top_layout.addWidget(self.lbl_iou_val)

        root_layout.addWidget(top_group)

        # ===================== 中部区域 =====================
        mid_splitter = QSplitter(Qt.Horizontal)

        # ---- 左侧：视频显示区域 (70%) ----
        self.lbl_display = QLabel("等待视频流或图片...")
        self.lbl_display.setObjectName("lblDisplay")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mid_splitter.addWidget(self.lbl_display)

        # ---- 右侧：操作与状态面板 (30%) ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        # 视频源选择
        src_group = QGroupBox("检测源")
        src_layout = QVBoxLayout(src_group)
        self.cmb_source = QComboBox()
        self.cmb_source.addItems([
            "摄像头 (0)",
            "摄像头 (1)",
            "视频文件...",
            "图片文件...",
        ])
        self.btn_select_image = QPushButton("选择图片")
        self.btn_select_image.setEnabled(False)
        src_layout.addWidget(self.cmb_source)
        src_layout.addWidget(self.btn_select_image)
        right_layout.addWidget(src_group)

        # 控制按钮
        ctrl_group = QGroupBox("检测控制")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.btn_start = QPushButton("▶  开始检测")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setMinimumHeight(42)
        self.btn_stop = QPushButton("■  停止检测")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_unload = QPushButton("卸载模型")
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_unload)
        right_layout.addWidget(ctrl_group)

        # 实时状态
        status_group = QGroupBox("实时状态")
        status_layout = QVBoxLayout(status_group)
        self.lbl_device = QLabel("设备: --")
        self.lbl_device.setStyleSheet("color: #f9e2af; font-weight: bold;")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setFont(QFont("Consolas", 14, QFont.Bold))
        self.lbl_objs = QLabel("检测目标: 0")
        self.lbl_objs.setFont(QFont("Consolas", 12))
        self.lbl_latency = QLabel("延迟: 0 ms")
        self.lbl_latency.setFont(QFont("Consolas", 12))
        status_layout.addWidget(self.lbl_device)
        status_layout.addWidget(self._separator())
        status_layout.addWidget(self.lbl_fps)
        status_layout.addWidget(self.lbl_objs)
        status_layout.addWidget(self.lbl_latency)
        right_layout.addWidget(status_group)

        right_layout.addStretch()
        mid_splitter.addWidget(right_panel)

        # 设置分割比例 7:3
        mid_splitter.setSizes([950, 410])
        root_layout.addWidget(mid_splitter, stretch=1)

        # ===================== 底部日志 =====================
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(140)
        log_layout.addWidget(self.txt_log)
        root_layout.addWidget(log_group)

    # ------------------------------------------------------------------
    #  辅助 UI 方法
    # ------------------------------------------------------------------
    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #45475a;")
        return line

    def update_frame(self, image_np):
        """
        将 numpy BGR 图像转换为 QPixmap 并显示。
        Ultralytics result.plot() 返回 BGR 格式。
        """
        if image_np is None or image_np.size == 0:
            return
        try:
            h, w, ch = image_np.shape
            bytes_per_line = ch * w
            # OpenCV BGR → Qt RGB: 使用 Format_BGR888 (PySide6 支持) 或 rgbSwapped
            qt_img = QImage(image_np.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(
                self.lbl_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_display.setPixmap(scaled)
        except Exception as e:
            print(f"帧显示错误: {e}")

    def update_status_info(self, fps: float, obj_count: int, latency_ms: float = 0):
        """更新右侧状态栏"""
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_objs.setText(f"检测目标: {obj_count}")
        if latency_ms > 0:
            self.lbl_latency.setText(f"延迟: {latency_ms:.0f} ms")
        else:
            computed = 1000.0 / fps if fps > 0 else 0
            self.lbl_latency.setText(f"延迟: {computed:.0f} ms")

    def set_device_label(self, device: str):
        """设置设备显示标签"""
        display = "GPU (CUDA)" if device == "cuda" else "CPU"
        self.lbl_device.setText(f"设备: {display}")

    def log(self, message: str):
        """向日志框追加带时间戳的消息"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}]  {message}")
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())
