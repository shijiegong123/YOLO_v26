"""
训练对话框 & 导出对话框
包含 TrainingDialog（模型训练）和 ExportDialog（模型导出）两个 QDialog。
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit,
    QGroupBox, QProgressBar, QCheckBox, QFileDialog, QTabWidget, QWidget,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from yolo_detector import OFFICIAL_DATASETS, EXPORT_FORMATS


# -----------------------------------------------------------------------
#  对话框通用样式（继承主窗口深色主题变量）
# -----------------------------------------------------------------------
DIALOG_STYLE = """
QDialog {
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
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #1e1e2e; color: #585b70; border-color: #313244; }
QPushButton#btnPrimary {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#btnPrimary:hover { background-color: #b4befe; }
QPushButton#btnDanger {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#btnDanger:hover { background-color: #eba0ac; }
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
QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QLineEdit:focus { border-color: #89b4fa; }
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 6px;
    background-color: #313244;
    text-align: center;
    color: #1e1e2e;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 5px;
}
QCheckBox {
    color: #cdd6f4;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1e1e2e;
    border-bottom: 2px solid #89b4fa;
    color: #89b4fa;
}
QTabBar::tab:hover { background: #45475a; }
"""


# =======================================================================
#  训练对话框
# =======================================================================
class TrainingDialog(QDialog):
    """
    模型训练对话框。
    支持选择官方数据集 / 自定义数据集 YAML，配置训练超参数，实时查看训练进度。
    """

    sig_start_training = Signal(dict)  # 发射训练参数字典给控制器

    def __init__(self, current_model_path: str = "", device: str = "cpu", parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型训练")
        self.setGeometry(120, 80, 720, 700)
        self.setStyleSheet(DIALOG_STYLE)
        self._current_model = current_model_path
        self._device = device
        self._training_thread = None
        self._custom_pretrained_path = ""  # 用户自定义选择的预训练模型路径
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        tabs = QTabWidget()

        # =================== Tab 1: 数据集 ===================
        tab_data = QWidget()
        tab_data_layout = QVBoxLayout(tab_data)

        # 官方数据集
        official_group = QGroupBox("官方数据集")
        official_layout = QVBoxLayout(official_group)
        self.cmb_official = QComboBox()
        self.cmb_official.addItem("-- 选择官方数据集 --")
        for name, info in OFFICIAL_DATASETS.items():
            self.cmb_official.addItem(f"{name}  [{info['desc']}]")
        official_layout.addWidget(self.cmb_official)
        tab_data_layout.addWidget(official_group)

        # 自定义数据集
        custom_group = QGroupBox("自定义数据集 (YOLO 格式)")
        custom_layout = QVBoxLayout(custom_group)
        hint = QLabel(
            "请提供 YOLO 格式的 data.yaml 文件路径。\n"
            "YAML 格式示例:\n"
            "  path: /dataset/root\n"
            "  train: images/train\n"
            "  val: images/val\n"
            "  names: [cat, dog, ...]"
        )
        hint.setStyleSheet("color: #a6adc8; font-size: 11px;")
        hint.setWordWrap(True)

        file_layout = QHBoxLayout()
        self.txt_custom_data = QLineEdit()
        self.txt_custom_data.setPlaceholderText("data.yaml 文件路径...")
        btn_browse_data = QPushButton("浏览...")
        btn_browse_data.clicked.connect(self._browse_dataset)
        file_layout.addWidget(self.txt_custom_data)
        file_layout.addWidget(btn_browse_data)

        custom_layout.addWidget(hint)
        custom_layout.addLayout(file_layout)
        tab_data_layout.addWidget(custom_group)
        tab_data_layout.addStretch()
        tabs.addTab(tab_data, "数据集")

        # =================== Tab 2: 训练参数 ===================
        tab_params = QWidget()
        params_layout = QGridLayout(tab_params)
        params_layout.setSpacing(10)

        row = 0
        # 预训练模型
        params_layout.addWidget(QLabel("预训练模型:"), row, 0)
        self.cmb_pretrained = QComboBox()
        self.cmb_pretrained.addItem("使用当前加载的模型")
        for name, info in [
            ("yolo26n.pt", "Nano"), ("yolo26s.pt", "Small"),
            ("yolo26m.pt", "Medium"), ("yolo26l.pt", "Large"),
            ("yolo26x.pt", "XLarge"),
            ("yolo11n.pt", "YOLO11-Nano"),
        ]:
            self.cmb_pretrained.addItem(f"{name} ({info})")
        self.cmb_pretrained.addItem("浏览自定义模型...")
        self.cmb_pretrained.setMinimumWidth(220)
        self.cmb_pretrained.currentIndexChanged.connect(self._on_pretrained_changed)
        params_layout.addWidget(self.cmb_pretrained, row, 1)

        row += 1
        # 任务类型
        params_layout.addWidget(QLabel("任务类型:"), row, 0)
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["detect", "segment", "pose", "obb", "classify"])
        params_layout.addWidget(self.cmb_task, row, 1)

        row += 1
        # Epochs
        params_layout.addWidget(QLabel("训练轮数 (Epochs):"), row, 0)
        self.spn_epochs = QSpinBox()
        self.spn_epochs.setRange(1, 10000)
        self.spn_epochs.setValue(100)
        params_layout.addWidget(self.spn_epochs, row, 1)

        row += 1
        # Image Size
        params_layout.addWidget(QLabel("图像尺寸 (ImgSz):"), row, 0)
        self.cmb_imgsz = QComboBox()
        self.cmb_imgsz.addItems(["320", "416", "512", "640", "800", "1024", "1280"])
        self.cmb_imgsz.setCurrentText("640")
        params_layout.addWidget(self.cmb_imgsz, row, 1)

        row += 1
        # Batch Size
        params_layout.addWidget(QLabel("批大小 (Batch):"), row, 0)
        self.spn_batch = QSpinBox()
        self.spn_batch.setRange(-1, 256)
        self.spn_batch.setValue(16)
        self.spn_batch.setSpecialValueText("Auto")
        params_layout.addWidget(self.spn_batch, row, 1)

        row += 1
        # 设备
        params_layout.addWidget(QLabel("训练设备:"), row, 0)
        self.cmb_device = QComboBox()
        self.cmb_device.addItem(f"自动 ({self._device})")
        self.cmb_device.addItem("cpu")
        if self._device == "cuda":
            self.cmb_device.addItem("cuda:0")
        params_layout.addWidget(self.cmb_device, row, 1)

        row += 1
        # 项目名称
        params_layout.addWidget(QLabel("实验名称:"), row, 0)
        self.txt_name = QLineEdit("yolo_train")
        params_layout.addWidget(self.txt_name, row, 1)

        row += 1
        # 输出目录
        params_layout.addWidget(QLabel("输出目录:"), row, 0)
        self.txt_project = QLineEdit("runs/train")
        params_layout.addWidget(self.txt_project, row, 1)

        row += 1
        # Workers
        params_layout.addWidget(QLabel("Workers:"), row, 0)
        self.spn_workers = QSpinBox()
        self.spn_workers.setRange(0, 32)
        self.spn_workers.setValue(8)
        params_layout.addWidget(self.spn_workers, row, 1)

        row += 1
        # 优化器
        params_layout.addWidget(QLabel("优化器:"), row, 0)
        self.cmb_optimizer = QComboBox()
        self.cmb_optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        params_layout.addWidget(self.cmb_optimizer, row, 1)

        row += 1
        # Patience (早停)
        params_layout.addWidget(QLabel("早停 Patience:"), row, 0)
        self.spn_patience = QSpinBox()
        self.spn_patience.setRange(0, 500)
        self.spn_patience.setValue(50)
        params_layout.addWidget(self.spn_patience, row, 1)

        row += 1
        params_layout.setRowStretch(row, 1)
        tabs.addTab(tab_params, "训练参数")

        layout.addWidget(tabs)

        # =================== 进度区域 ===================
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.lbl_epoch_info = QLabel("Epoch: --/--")
        self.lbl_epoch_info.setFont(QFont("Consolas", 12))
        progress_layout.addWidget(self.lbl_epoch_info)

        self.txt_train_log = QTextEdit()
        self.txt_train_log.setReadOnly(True)
        self.txt_train_log.setMaximumHeight(160)
        progress_layout.addWidget(self.txt_train_log)

        layout.addWidget(progress_group)

        # =================== 按钮区域 ===================
        btn_layout = QHBoxLayout()
        self.btn_start_train = QPushButton("开始训练")
        self.btn_start_train.setObjectName("btnPrimary")
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.setObjectName("btnDanger")
        self.btn_stop_train.setEnabled(False)
        self.btn_close = QPushButton("关闭")

        btn_layout.addWidget(self.btn_start_train)
        btn_layout.addWidget(self.btn_stop_train)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        # 连接内部信号
        self.btn_close.clicked.connect(self.close)

    # ------------------------------------------------------------------
    def _on_pretrained_changed(self, index: int):
        """当预训练模型下拉框变化时，若选中最后一项则弹出文件选择"""
        if index == self.cmb_pretrained.count() - 1:  # 最后一项 = 浏览自定义
            path, _ = QFileDialog.getOpenFileName(
                self, "选择预训练模型文件", "", "PyTorch Models (*.pt)"
            )
            if path:
                self._custom_pretrained_path = path
                # 替换显示文本（不触发信号）
                self.cmb_pretrained.blockSignals(True)
                self.cmb_pretrained.setItemText(index, f"自定义: {os.path.basename(path)}")
                self.cmb_pretrained.blockSignals(False)
            else:
                self.cmb_pretrained.blockSignals(True)
                self.cmb_pretrained.setCurrentIndex(0)
                self.cmb_pretrained.blockSignals(False)

    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集 YAML 文件", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.txt_custom_data.setText(path)
            self.cmb_official.setCurrentIndex(0)  # 清除官方选择

    def get_train_params(self) -> dict:
        """收集当前 UI 中的训练参数，返回字典。"""
        # 数据集来源
        custom_data = self.txt_custom_data.text().strip()
        official_idx = self.cmb_official.currentIndex()

        if custom_data:
            data = custom_data
        elif official_idx > 0:
            keys = list(OFFICIAL_DATASETS.keys())
            data = OFFICIAL_DATASETS[keys[official_idx - 1]]["data"]
        else:
            data = ""

        # 预训练模型
        pretrained_idx = self.cmb_pretrained.currentIndex()
        if pretrained_idx == 0:
            pretrained = self._current_model  # 使用当前模型
        elif pretrained_idx == self.cmb_pretrained.count() - 1 or self._custom_pretrained_path:
            pretrained = self._custom_pretrained_path  # 自定义浏览的模型
        else:
            pretrained = self.cmb_pretrained.currentText().split(" ")[0]

        # 设备
        device_text = self.cmb_device.currentText()
        if "自动" in device_text:
            device = self._device
        elif device_text == "cpu":
            device = "cpu"
        else:
            device = device_text

        return {
            "data": data,
            "pretrained": pretrained,
            "task": self.cmb_task.currentText(),
            "epochs": self.spn_epochs.value(),
            "imgsz": int(self.cmb_imgsz.currentText()),
            "batch": self.spn_batch.value(),
            "device": device,
            "project": self.txt_project.text().strip() or "runs/train",
            "name": self.txt_name.text().strip() or "yolo_train",
            "extra_args": {
                "workers": self.spn_workers.value(),
                "optimizer": self.cmb_optimizer.currentText(),
                "patience": self.spn_patience.value(),
            },
        }

    # ------------------------------------------------------------------
    #  外部调用的 UI 更新方法
    # ------------------------------------------------------------------
    def append_log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_train_log.append(f"[{ts}] {msg}")
        sb = self.txt_train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_progress(self, percent: float):
        self.progress_bar.setValue(int(percent))

    def update_epoch(self, current: int, total: int):
        self.lbl_epoch_info.setText(f"Epoch: {current}/{total}")

    def set_training_state(self, is_training: bool):
        self.btn_start_train.setEnabled(not is_training)
        self.btn_stop_train.setEnabled(is_training)


# =======================================================================
#  导出对话框
# =======================================================================
class ExportDialog(QDialog):
    """
    模型导出对话框。
    支持选择导出格式、配置参数，实时查看导出日志。
    """

    sig_start_export = Signal(dict)  # 发射导出参数字典给控制器

    def __init__(self, current_model_path: str = "", device: str = "cpu", parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型导出")
        self.setGeometry(150, 120, 560, 480)
        self.setStyleSheet(DIALOG_STYLE)
        self._current_model = current_model_path
        self._device = device
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ---- 模型选择 ----
        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout(model_group)
        self.lbl_model = QLabel(f"当前模型: {self._current_model or '未加载'}")
        self.lbl_model.setStyleSheet("color: #f9e2af;")
        model_layout.addWidget(self.lbl_model)

        file_layout = QHBoxLayout()
        self.txt_model_path = QLineEdit(self._current_model)
        self.txt_model_path.setPlaceholderText("模型文件路径...")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_model)
        file_layout.addWidget(self.txt_model_path)
        file_layout.addWidget(btn_browse)
        model_layout.addLayout(file_layout)
        layout.addWidget(model_group)

        # ---- 导出参数 ----
        param_group = QGroupBox("导出参数")
        param_layout = QGridLayout(param_group)

        row = 0
        param_layout.addWidget(QLabel("导出格式:"), row, 0)
        self.cmb_format = QComboBox()
        for name, info in EXPORT_FORMATS.items():
            self.cmb_format.addItem(f"{name}  — {info['desc']}")
        self.cmb_format.setCurrentIndex(1)  # 默认 ONNX
        param_layout.addWidget(self.cmb_format, row, 1)

        row += 1
        param_layout.addWidget(QLabel("图像尺寸:"), row, 0)
        self.cmb_imgsz = QComboBox()
        self.cmb_imgsz.addItems(["320", "416", "512", "640", "800", "1024"])
        self.cmb_imgsz.setCurrentText("640")
        param_layout.addWidget(self.cmb_imgsz, row, 1)

        row += 1
        param_layout.addWidget(QLabel("ONNX Opset:"), row, 0)
        self.spn_opset = QSpinBox()
        self.spn_opset.setRange(9, 20)
        self.spn_opset.setValue(12)
        param_layout.addWidget(self.spn_opset, row, 1)

        row += 1
        self.chk_half = QCheckBox("FP16 半精度 (仅 GPU)")
        param_layout.addWidget(self.chk_half, row, 0, 1, 2)

        row += 1
        self.chk_simplify = QCheckBox("简化 ONNX 模型")
        self.chk_simplify.setChecked(True)
        param_layout.addWidget(self.chk_simplify, row, 0, 1, 2)

        row += 1
        self.chk_dynamic = QCheckBox("动态输入尺寸")
        param_layout.addWidget(self.chk_dynamic, row, 0, 1, 2)

        layout.addWidget(param_group)

        # ---- 日志 ----
        log_group = QGroupBox("导出日志")
        log_layout = QVBoxLayout(log_group)
        self.txt_export_log = QTextEdit()
        self.txt_export_log.setReadOnly(True)
        self.txt_export_log.setMaximumHeight(120)
        log_layout.addWidget(self.txt_export_log)
        layout.addWidget(log_group)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("开始导出")
        self.btn_export.setObjectName("btnPrimary")
        self.btn_close = QPushButton("关闭")
        btn_layout.addWidget(self.btn_export)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.btn_close.clicked.connect(self.close)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch Models (*.pt)"
        )
        if path:
            self.txt_model_path.setText(path)

    def get_export_params(self) -> dict:
        """收集导出参数。"""
        format_idx = self.cmb_format.currentIndex()
        format_keys = list(EXPORT_FORMATS.keys())
        fmt = EXPORT_FORMATS[format_keys[format_idx]]["format"]

        return {
            "model_path": self.txt_model_path.text().strip(),
            "format": fmt,
            "imgsz": int(self.cmb_imgsz.currentText()),
            "half": self.chk_half.isChecked(),
            "simplify": self.chk_simplify.isChecked(),
            "opset": self.spn_opset.value(),
            "dynamic": self.chk_dynamic.isChecked(),
            "device": self._device,
        }

    def append_log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_export_log.append(f"[{ts}] {msg}")
        sb = self.txt_export_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_exporting_state(self, is_exporting: bool):
        self.btn_export.setEnabled(not is_exporting)
