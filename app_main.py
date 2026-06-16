import sys
import os
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtCore import Slot

from main_window import MainWindow
from worker_thread import DetectionThread, ImageDetectionThread
from yolo_detector import YOLO_Detector
from utils import get_builtin_models


class YOLOAppController:
    """应用控制器：连接 GUI 与检测逻辑"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.detector = YOLO_Detector()
        self.worker_thread: DetectionThread | None = None
        self.image_thread: ImageDetectionThread | None = None
        self._builtin_models = get_builtin_models()  # (name, file, task, desc)
        self._selected_image_path: str | None = None

        self._setup_connections()
        self.window.show()
        self.window.set_device_label(self.detector.device)
        self.window.log("欢迎使用 Ultralytics YOLO 智能检测平台")
        self.window.log(f"当前设备: {self.detector.device}")

    # ------------------------------------------------------------------
    #  信号槽连接
    # ------------------------------------------------------------------
    def _setup_connections(self):
        w = self.window
        w.btn_load_model.clicked.connect(self._on_load_custom_model)
        w.cmb_builtin.currentIndexChanged.connect(self._on_builtin_model_selected)
        w.btn_start.clicked.connect(self._on_start_detection)
        w.btn_stop.clicked.connect(self._on_stop_detection)
        w.btn_unload.clicked.connect(self._on_unload_model)
        w.sld_conf.valueChanged.connect(self._on_conf_changed)
        w.sld_iou.valueChanged.connect(self._on_iou_changed)
        w.cmb_source.currentTextChanged.connect(self._on_source_changed)
        w.btn_select_image.clicked.connect(self._on_select_image)
        self.app.aboutToQuit.connect(self._cleanup)

    # ------------------------------------------------------------------
    #  模型加载
    # ------------------------------------------------------------------
    @Slot()
    def _on_load_custom_model(self):
        """用户点击'加载模型'按钮，选择本地 .pt 文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "选择 YOLO 模型文件", "", "PyTorch Models (*.pt)"
        )
        if not file_path:
            return

        task_type = self.window.cmb_task.currentText()
        self._do_load_model(file_path, task_type)

    @Slot(int)
    def _on_builtin_model_selected(self, index: int):
        """用户从内置模型下拉框选择模型"""
        if index <= 0:
            return  # 占位项
        model_info = self._builtin_models[index - 1]  # (name, file, task, desc)
        _name, model_file, task, _desc = model_info

        # 自动切换任务类型
        task_idx = self.window.cmb_task.findText(task)
        if task_idx >= 0:
            self.window.cmb_task.blockSignals(True)
            self.window.cmb_task.setCurrentIndex(task_idx)
            self.window.cmb_task.blockSignals(False)

        self._do_load_model(model_file, task)

    def _do_load_model(self, model_path: str, task_type: str):
        """执行模型加载（主线程中）"""
        self.window.log(f"正在加载模型: {model_path} ...")

        # 设置下载回调
        self.detector.set_download_callback(lambda msg: self.window.log(msg))
        success = self.detector.load_model(model_path, task_type)

        if success:
            self.window.lbl_model_path.setText(os.path.basename(model_path))
            task_cn = YOLO_Detector.get_task_display_name(self.detector.task_type)
            self.window.log(
                f"模型加载成功: {model_path}  |  任务: {task_cn}  |  "
                f"设备: {self.detector.device}"
            )
        else:
            QMessageBox.critical(self.window, "错误", "模型加载失败，请检查文件或网络。")
            self.window.log("模型加载失败")

    @Slot()
    def _on_unload_model(self):
        """卸载当前模型"""
        self._stop_current_thread()
        self.detector.unload_model()
        self.window.lbl_model_path.setText("未选择模型")
        self.window.cmb_builtin.blockSignals(True)
        self.window.cmb_builtin.setCurrentIndex(0)
        self.window.cmb_builtin.blockSignals(False)
        self.window.log("模型已卸载")

    # ------------------------------------------------------------------
    #  检测控制
    # ------------------------------------------------------------------
    @Slot()
    def _on_start_detection(self):
        if not self.detector.is_loaded():
            QMessageBox.warning(self.window, "提示", "请先加载模型！")
            return

        source_text = self.window.cmb_source.currentText()
        conf = self.window.sld_conf.value() / 100.0
        iou = self.window.sld_iou.value() / 100.0
        task_type = self.window.cmb_task.currentText()
        model_path = self.detector.current_model_path
        device = self.detector.device

        # ---------- 图片模式 ----------
        if "图片文件" in source_text:
            if not self._selected_image_path:
                self._on_select_image()
                if not self._selected_image_path:
                    return
            self._start_image_detection(
                self._selected_image_path, model_path, task_type, conf, iou, device
            )
            return

        # ---------- 视频 / 摄像头模式 ----------
        source = self._resolve_video_source(source_text)
        if source is None:
            return  # 用户取消了文件选择

        self._stop_current_thread()

        self.worker_thread = DetectionThread(
            source=source,
            model_path=model_path,
            task_type=task_type,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
        )
        self.worker_thread.sig_frame_processed.connect(self.window.update_frame)
        self.worker_thread.sig_status_update.connect(self.window.update_status_info)
        self.worker_thread.sig_error.connect(self._on_thread_error)
        self.worker_thread.sig_detection_done.connect(self._on_thread_finished)
        self.worker_thread.start()

        self.window.btn_start.setEnabled(False)
        self.window.btn_stop.setEnabled(True)
        self.window.log(f"开始视频检测 — 源: {source}, Conf: {conf:.2f}, IOU: {iou:.2f}")

    def _start_image_detection(self, image_path, model_path, task_type, conf, iou, device):
        """启动单张图片检测线程"""
        self._stop_current_thread()

        self.image_thread = ImageDetectionThread(
            image_path=image_path,
            model_path=model_path,
            task_type=task_type,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
        )
        self.image_thread.sig_image_ready.connect(self.window.update_frame)
        self.image_thread.sig_status_update.connect(self.window.update_status_info)
        self.image_thread.sig_error.connect(self._on_thread_error)
        self.image_thread.sig_done.connect(self._on_image_done)
        self.image_thread.start()

        self.window.log(f"开始图片检测 — {image_path}")

    @Slot()
    def _on_stop_detection(self):
        self._stop_current_thread()
        self.window.log("正在停止检测...")

    def _stop_current_thread(self):
        """停止当前正在运行的工作线程"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(3000)
            self.worker_thread = None
        if self.image_thread and self.image_thread.isRunning():
            self.image_thread.wait(3000)
            self.image_thread = None

    # ------------------------------------------------------------------
    #  线程回调
    # ------------------------------------------------------------------
    @Slot()
    def _on_thread_finished(self):
        self.window.btn_start.setEnabled(True)
        self.window.btn_stop.setEnabled(False)
        self.window.lbl_display.setText("检测已结束")
        self.window.log("视频检测已停止，资源已释放")

    @Slot()
    def _on_image_done(self):
        self.window.btn_start.setEnabled(True)
        self.window.btn_stop.setEnabled(False)
        self.window.log("图片检测完成")

    @Slot(str)
    def _on_thread_error(self, msg: str):
        self.window.log(f"[错误] {msg}")
        QMessageBox.warning(self.window, "运行错误", msg)

    # ------------------------------------------------------------------
    #  参数变更
    # ------------------------------------------------------------------
    @Slot(int)
    def _on_conf_changed(self, value: int):
        self.window.lbl_conf_val.setText(f"Conf: {value / 100:.2f}")

    @Slot(int)
    def _on_iou_changed(self, value: int):
        self.window.lbl_iou_val.setText(f"IOU: {value / 100:.2f}")

    # ------------------------------------------------------------------
    #  源选择
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_source_changed(self, text: str):
        self.window.btn_select_image.setEnabled("图片文件" in text)
        if "图片文件" not in text:
            self._selected_image_path = None

    @Slot()
    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "选择图片文件", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if path:
            self._selected_image_path = path
            self.window.log(f"已选择图片: {path}")

    # ------------------------------------------------------------------
    #  辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_video_source(text: str):
        """根据下拉框文本解析视频源。返回 int(摄像头) 或 str(文件路径) 或 None"""
        if "视频文件" in text:
            path, _ = QFileDialog.getOpenFileName(
                None, "选择视频文件", "",
                "Video (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
            )
            return path if path else None

        # 摄像头，提取编号
        try:
            cam_id = int(text.split("(")[1].split(")")[0])
        except (IndexError, ValueError):
            cam_id = 0
        return cam_id

    # ------------------------------------------------------------------
    #  清理 & 运行
    # ------------------------------------------------------------------
    def _cleanup(self):
        self._stop_current_thread()
        self.window.log("程序已退出")

    def exec_app(self):
        sys.exit(self.app.exec())


# ======================================================================
if __name__ == "__main__":
    controller = YOLOAppController()
    controller.exec_app()
