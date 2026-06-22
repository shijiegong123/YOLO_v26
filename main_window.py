import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QTextEdit, QGroupBox, QSizePolicy, QSplitter,
    QFrame, QToolButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QShortcut, QKeySequence
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
    border-radius: 6px;
    margin-top: 6px;
    padding: 8px 6px 4px 6px;
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
    padding: 4px 12px;
    min-height: 24px;
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
    padding: 3px 6px;
    min-height: 22px;
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
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 2px 4px;
    padding-right: 22px;
    color: #cdd6f4;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #45475a;
    border-bottom: 1px solid #45475a;
    border-top-right-radius: 4px;
    background: #313244;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #45475a;
    border-bottom-right-radius: 4px;
    background: #313244;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #45475a;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px; height: 8px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px; height: 8px;
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
    min-width: 200px;
    min-height: 100px;
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
        self.setGeometry(80, 60, 1200, 900)
        self.setMinimumSize(1200, 900)
        self.setStyleSheet(DARK_STYLE)
        self._builtin_models = get_builtin_models()
        self._current_pixmap: QPixmap | None = None
        self.init_ui()

    # ------------------------------------------------------------------
    #  UI 构建
    # ------------------------------------------------------------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(4)

        # ===================== 顶部工具栏 =====================
        top_group = QGroupBox("模型与参数控制")
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(8)

        # 加载自定义模型
        self.btn_load_model = QPushButton("加载模型 (.pt)")
        self.btn_load_model.setMinimumWidth(100)

        # 内置模型下拉框
        self.cmb_builtin = QComboBox()
        self.cmb_builtin.addItem("-- 选择内置模型 --")
        for name, _file, _task, desc in self._builtin_models:
            self.cmb_builtin.addItem(f"{name}  [{desc}]")
        self.cmb_builtin.setMinimumWidth(220)

        # 模型路径标签
        self.lbl_model_path = QLabel("未选择模型")
        self.lbl_model_path.setObjectName("lblModelPath")
        self.lbl_model_path.setMinimumWidth(100)
        self.lbl_model_path.setMaximumWidth(180)

        # 任务类型
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["detect", "segment", "pose", "obb"])
        self.cmb_task.setMinimumWidth(80)

        # 置信度滑块
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(5, 95)
        self.sld_conf.setValue(25)
        self.sld_conf.setMinimumWidth(100)
        self.lbl_conf_val = QLabel("Conf: 0.25")
        self.lbl_conf_val.setMinimumWidth(65)

        # IOU 滑块
        self.sld_iou = QSlider(Qt.Horizontal)
        self.sld_iou.setRange(10, 95)
        self.sld_iou.setValue(45)
        self.sld_iou.setMinimumWidth(100)
        self.lbl_iou_val = QLabel("IOU: 0.45")
        self.lbl_iou_val.setMinimumWidth(65)

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
        right_layout.setSpacing(2)

        # 视频源选择
        src_group = QGroupBox("检测源")
        src_layout = QVBoxLayout(src_group)
        self.cmb_source = QComboBox()
        self.cmb_source.addItems([
            "摄像头 (0)",
            "摄像头 (1)",
            "视频文件...",
            "图片文件...",
            "图片文件夹...",
        ])
        self.btn_select_image = QPushButton("选择图片")
        self.btn_select_image.setEnabled(False)
        self.btn_select_folder = QPushButton("选择文件夹")
        self.btn_select_folder.setEnabled(False)
        src_layout.addWidget(self.cmb_source)
        src_layout.addWidget(self.btn_select_image)
        src_layout.addWidget(self.btn_select_folder)
        right_layout.addWidget(src_group)

        # 控制按钮
        ctrl_group = QGroupBox("检测控制")
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.btn_start = QPushButton("▶  开始检测")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setMinimumHeight(28)
        self.btn_stop = QPushButton("■  停止检测")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setMinimumHeight(28)
        self.btn_stop.setEnabled(False)
        self.btn_unload = QPushButton("卸载模型")
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_unload)

        # 保存结果（手动按钮）
        self.btn_save_result = QPushButton("💾  保存当前结果")
        self.btn_save_result.setEnabled(False)
        ctrl_layout.addWidget(self.btn_save_result)

        # 类别过滤
        self.btn_class_filter = QPushButton("类别过滤")
        ctrl_layout.addWidget(self.btn_class_filter)

        right_layout.addWidget(ctrl_group)

        # 工具按钮组
        tool_group = QGroupBox("工具")
        tool_layout = QVBoxLayout(tool_group)
        self.btn_train = QPushButton("\U0001F3CB  模型训练")
        self.btn_train.setMinimumHeight(26)
        self.btn_export = QPushButton("\U0001F4E6  模型导出")
        self.btn_export.setMinimumHeight(26)
        self.btn_history = QPushButton("\U0001F4CB  历史记录")
        self.btn_history.setMinimumHeight(26)
        tool_layout.addWidget(self.btn_train)
        tool_layout.addWidget(self.btn_export)
        tool_layout.addWidget(self.btn_history)
        right_layout.addWidget(tool_group)

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

        # 设置分割比例 60:40
        mid_splitter.setSizes([800, 400])

        # ===================== 底部日志 =====================
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)

        # 使用垂直 Splitter 让中部和日志区可拖拽调整比例
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(mid_splitter)
        vsplit.addWidget(log_group)
        vsplit.setSizes([450, 130])
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        root_layout.addWidget(vsplit, stretch=1)

        # ===================== 快捷键 =====================
        self._setup_shortcuts()

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
        存储 pixmap 以便 resize 时重新缩放。
        """
        if image_np is None or image_np.size == 0:
            return
        try:
            h, w, ch = image_np.shape
            bytes_per_line = ch * w
            qt_img = QImage(image_np.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self._current_pixmap = QPixmap.fromImage(qt_img)
            self._show_scaled_pixmap()
        except Exception as e:
            print(f"帧显示错误: {e}")

    def _show_scaled_pixmap(self):
        """将当前存储的 pixmap 按 label 尺寸等比缩放后显示"""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        lbl_size = self.lbl_display.size()
        if lbl_size.width() <= 0 or lbl_size.height() <= 0:
            return
        scaled = self._current_pixmap.scaled(
            lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_display.setPixmap(scaled)

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图片"""
        super().resizeEvent(event)
        self._show_scaled_pixmap()

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

    # ------------------------------------------------------------------
    #  快捷键设置
    # ------------------------------------------------------------------
    def _setup_shortcuts(self):
        """设置全局快捷键"""
        shortcuts = [
            ("Ctrl+O", self.btn_load_model.click),
            ("Ctrl+S", self.btn_start.click),
            ("Ctrl+Q", self.btn_stop.click),
            ("Ctrl+T", self.btn_train.click),
            ("Ctrl+E", self.btn_export.click),
            ("Ctrl+H", self.btn_history.click),
            ("Space",  self._toggle_detection),
            ("F11",    self._toggle_fullscreen),
        ]
        for key_seq, slot in shortcuts:
            sc = QShortcut(QKeySequence(key_seq), self)
            sc.activated.connect(slot)

    def _toggle_detection(self):
        """Space 键切换开始/停止检测"""
        if self.btn_start.isEnabled():
            self.btn_start.click()
        elif self.btn_stop.isEnabled():
            self.btn_stop.click()

    def _toggle_fullscreen(self):
        """F11 切换全屏"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
