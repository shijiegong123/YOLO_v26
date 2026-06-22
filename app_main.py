import sys
import os
import cv2
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QDialog, QListWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Slot

from main_window import MainWindow
from worker_thread import DetectionThread, ImageDetectionThread, TrainingThread, ExportThread, BatchImageThread
from yolo_detector import YOLO_Detector
from utils import get_builtin_models
from train_dialog import TrainingDialog, ExportDialog
from history_dialog import HistoryDialog
import history


class YOLOAppController:
    """应用控制器：连接 GUI 与检测逻辑"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.detector = YOLO_Detector()
        self.worker_thread: DetectionThread | None = None
        self.image_thread: ImageDetectionThread | None = None
        self.training_thread: TrainingThread | None = None
        self.export_thread: ExportThread | None = None
        self._builtin_models = get_builtin_models()  # (name, file, task, desc)
        self._selected_image_path: str | None = None
        self._train_dialog: TrainingDialog | None = None
        self._export_dialog: ExportDialog | None = None
        self._history_dialog: HistoryDialog | None = None
        self._batch_thread: BatchImageThread | None = None
        self._selected_folder_path: str | None = None
        self._save_dir: str = ""
        self._allowed_classes: list | None = None  # None = 不过滤
        self._last_result_image = None  # 保存最近一次检测结果图像

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
        w.btn_select_folder.clicked.connect(self._on_select_folder)
        w.btn_train.clicked.connect(self._on_open_train_dialog)
        w.btn_export.clicked.connect(self._on_open_export_dialog)
        w.btn_history.clicked.connect(self._on_open_history_dialog)
        w.btn_class_filter.clicked.connect(self._on_class_filter)
        w.btn_save_result.clicked.connect(self._on_save_result)
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
        save_dir = self._save_dir if self._save_dir else ""
        allowed = self._allowed_classes

        # ---------- 图片文件夹批量模式 ----------
        if "图片文件夹" in source_text:
            if not self._selected_folder_path:
                self._on_select_folder()
                if not self._selected_folder_path:
                    return
            self._start_batch_detection(
                self._selected_folder_path, model_path, task_type, conf, iou, device, save_dir, allowed
            )
            return

        # ---------- 图片模式 ----------
        if "图片文件" in source_text:
            if not self._selected_image_path:
                self._on_select_image()
                if not self._selected_image_path:
                    return
            self._start_image_detection(
                self._selected_image_path, model_path, task_type, conf, iou, device, save_dir, allowed
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
            save_dir=save_dir,
            allowed_classes=allowed,
        )
        self.worker_thread.sig_frame_processed.connect(self._store_result_image)
        self.worker_thread.sig_status_update.connect(self.window.update_status_info)
        self.worker_thread.sig_error.connect(self._on_thread_error)
        self.worker_thread.sig_detection_done.connect(self._on_thread_finished)
        self.worker_thread.start()

        self.window.btn_start.setEnabled(False)
        self.window.btn_stop.setEnabled(True)
        self.window.log(f"开始视频检测 — 源: {source}, Conf: {conf:.2f}, IOU: {iou:.2f}")

    @Slot(object)
    def _store_result_image(self, image_np):
        """缓存最近一次检测结果图像，并转发给显示"""
        self._last_result_image = image_np.copy() if image_np is not None else None
        self.window.update_frame(image_np)
        if self._last_result_image is not None:
            self.window.btn_save_result.setEnabled(True)

    def _start_image_detection(self, image_path, model_path, task_type, conf, iou, device,
                                save_dir="", allowed_classes=None):
        """启动单张图片检测线程"""
        self._stop_current_thread()

        self.image_thread = ImageDetectionThread(
            image_path=image_path,
            model_path=model_path,
            task_type=task_type,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
            save_dir=save_dir,
            allowed_classes=allowed_classes,
        )
        self.image_thread.sig_image_ready.connect(self._store_result_image)
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
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.wait(3000)
            self._batch_thread = None

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
        # 写入历史记录
        self._record_detection("image")

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
        self.window.btn_select_folder.setEnabled("图片文件夹" in text)
        if "图片文件" not in text:
            self._selected_image_path = None
        if "图片文件夹" not in text:
            self._selected_folder_path = None

    @Slot()
    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "选择图片文件", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if path:
            self._selected_image_path = path
            self.window.log(f"已选择图片: {path}")

    @Slot()
    def _on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(self.window, "选择图片文件夹")
        if folder:
            self._selected_folder_path = folder
            self.window.log(f"已选择文件夹: {folder}")

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
    #  训练对话框
    # ------------------------------------------------------------------
    @Slot()
    def _on_open_train_dialog(self):
        """打开训练对话框"""
        model_path = self.detector.current_model_path or ""
        if self._train_dialog is None or not self._train_dialog.isVisible():
            self._train_dialog = TrainingDialog(
                current_model_path=model_path,
                device=self.detector.device,
                parent=self.window,
            )
            # 连接对话框按钮
            self._train_dialog.btn_start_train.clicked.connect(self._on_start_training)
            self._train_dialog.btn_stop_train.clicked.connect(self._on_stop_training)

        self._train_dialog.show()
        self._train_dialog.raise_()

    @Slot()
    def _on_start_training(self):
        """从训练对话框收集参数并启动训练线程"""
        if not self._train_dialog:
            return

        params = self._train_dialog.get_train_params()
        if not params["data"]:
            QMessageBox.warning(self._train_dialog, "提示", "请先选择或输入数据集！")
            return

        if not params["pretrained"]:
            QMessageBox.warning(self._train_dialog, "提示", "请先加载或选择预训练模型！")
            return

        # 停止之前的训练线程
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait(3000)

        self.training_thread = TrainingThread(
            pretrained_model=params["pretrained"],
            data=params["data"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            device=params["device"],
            project=params["project"],
            name=params["name"],
            task=params["task"],
            extra_args=params["extra_args"],
        )

        # 连接信号
        self.training_thread.sig_log.connect(self._on_train_log)
        self.training_thread.sig_epoch_done.connect(self._on_train_epoch)
        self.training_thread.sig_progress.connect(self._on_train_progress)
        self.training_thread.sig_finished.connect(self._on_train_finished)
        self.training_thread.sig_error.connect(self._on_train_error)

        self.training_thread.start()
        self._train_dialog.set_training_state(True)
        self.window.log(f"训练已开始 — 数据集: {params['data']}")

    @Slot()
    def _on_stop_training(self):
        """停止训练"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            if self._train_dialog:
                self._train_dialog.append_log("正在停止训练...")
            self.window.log("正在停止训练...")

    @Slot(str)
    def _on_train_log(self, msg: str):
        if self._train_dialog:
            self._train_dialog.append_log(msg)
        self.window.log(f"[训练] {msg}")

    @Slot(int, int)
    def _on_train_epoch(self, current: int, total: int):
        if self._train_dialog:
            self._train_dialog.update_epoch(current, total)

    @Slot(float)
    def _on_train_progress(self, percent: float):
        if self._train_dialog:
            self._train_dialog.update_progress(percent)

    @Slot(str)
    def _on_train_finished(self, best_path: str):
        if self._train_dialog:
            self._train_dialog.set_training_state(False)
            self._train_dialog.append_log(f"训练完成！最佳模型: {best_path}")
        self.window.log(f"训练完成！最佳模型保存至: {best_path}")
        QMessageBox.information(
            self.window, "训练完成",
            f"训练已完成！\n最佳模型保存至:\n{best_path}"
        )
        self.training_thread = None

    @Slot(str)
    def _on_train_error(self, msg: str):
        if self._train_dialog:
            self._train_dialog.append_log(f"[错误] {msg}")
            self._train_dialog.set_training_state(False)
        self.window.log(f"[训练错误] {msg}")
        QMessageBox.critical(self.window, "训练错误", msg)

    # ------------------------------------------------------------------
    #  导出对话框
    # ------------------------------------------------------------------
    @Slot()
    def _on_open_export_dialog(self):
        """打开导出对话框"""
        model_path = self.detector.current_model_path or ""
        if self._export_dialog is None or not self._export_dialog.isVisible():
            self._export_dialog = ExportDialog(
                current_model_path=model_path,
                device=self.detector.device,
                parent=self.window,
            )
            self._export_dialog.btn_export.clicked.connect(self._on_start_export)

        self._export_dialog.show()
        self._export_dialog.raise_()

    @Slot()
    def _on_start_export(self):
        """从导出对话框收集参数并启动导出线程"""
        if not self._export_dialog:
            return

        params = self._export_dialog.get_export_params()
        if not params["model_path"]:
            QMessageBox.warning(self._export_dialog, "提示", "请先选择或加载模型！")
            return

        # 停止之前的导出线程
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.wait(3000)

        self.export_thread = ExportThread(
            model_path=params["model_path"],
            export_format=params["format"],
            imgsz=params["imgsz"],
            half=params["half"],
            simplify=params["simplify"],
            opset=params["opset"],
            dynamic=params["dynamic"],
            device=params["device"],
        )

        self.export_thread.sig_log.connect(self._on_export_log)
        self.export_thread.sig_finished.connect(self._on_export_finished)
        self.export_thread.sig_error.connect(self._on_export_error)
        self.export_thread.sig_done.connect(self._on_export_done)

        self.export_thread.start()
        self._export_dialog.set_exporting_state(False)
        self.window.log(f"模型导出已开始 — 格式: {params['format']}")

    @Slot(str)
    def _on_export_log(self, msg: str):
        if self._export_dialog:
            self._export_dialog.append_log(msg)
        self.window.log(f"[导出] {msg}")

    @Slot(str)
    def _on_export_finished(self, exported_path: str):
        if self._export_dialog:
            self._export_dialog.append_log(f"导出完成: {exported_path}")
        self.window.log(f"模型导出完成: {exported_path}")
        QMessageBox.information(
            self.window, "导出完成",
            f"模型已成功导出至:\n{exported_path}"
        )

    @Slot(str)
    def _on_export_error(self, msg: str):
        if self._export_dialog:
            self._export_dialog.append_log(f"[错误] {msg}")
        self.window.log(f"[导出错误] {msg}")
        QMessageBox.critical(self.window, "导出错误", msg)

    @Slot()
    def _on_export_done(self):
        if self._export_dialog:
            self._export_dialog.set_exporting_state(False)
        self.export_thread = None

    # ------------------------------------------------------------------
    #  批量检测
    # ------------------------------------------------------------------
    def _start_batch_detection(self, folder_path, model_path, task_type, conf, iou, device,
                                save_dir="", allowed_classes=None):
        """启动批量图片检测"""
        self._stop_current_thread()

        self._batch_thread = BatchImageThread(
            folder_path=folder_path,
            model_path=model_path,
            task_type=task_type,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
            save_dir=save_dir,
            allowed_classes=allowed_classes,
        )
        self._batch_thread.sig_image_ready.connect(self._store_result_image)
        self._batch_thread.sig_status_update.connect(self._on_batch_status)
        self._batch_thread.sig_log.connect(lambda m: self.window.log(m))
        self._batch_thread.sig_error.connect(self._on_thread_error)
        self._batch_thread.sig_done.connect(self._on_batch_done)
        self._batch_thread.start()

        self.window.btn_start.setEnabled(False)
        self.window.btn_stop.setEnabled(True)
        self.window.log(f"批量检测已开始 — 文件夹: {folder_path}")

    @Slot(float, int, float)
    def _on_batch_status(self, progress: float, obj_count: int, latency_ms: float):
        self.window.lbl_fps.setText(f"进度: {progress:.0f}%")
        self.window.lbl_objs.setText(f"检测目标: {obj_count}")
        self.window.lbl_latency.setText(f"延迟: {latency_ms:.0f} ms")

    @Slot()
    def _on_batch_done(self):
        self.window.btn_start.setEnabled(True)
        self.window.btn_stop.setEnabled(False)
        self.window.log("批量检测已完成")
        self._record_detection("batch")
        self._batch_thread = None

    # ------------------------------------------------------------------
    #  保存结果（手动）
    # ------------------------------------------------------------------
    @Slot()
    def _on_save_result(self):
        """手动保存当前检测结果图像"""
        if self._last_result_image is None:
            QMessageBox.information(self.window, "提示", "没有可保存的检测结果")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.window, "保存检测结果", "detection_result.jpg",
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"
        )
        if save_path:
            try:
                cv2.imwrite(save_path, self._last_result_image)
                self.window.log(f"结果已保存至: {save_path}")
            except Exception as e:
                QMessageBox.critical(self.window, "保存失败", str(e))

    # ------------------------------------------------------------------
    #  类别过滤
    # ------------------------------------------------------------------
    @Slot()
    def _on_class_filter(self):
        """弹出类别选择对话框"""
        if not self.detector.is_loaded():
            QMessageBox.warning(self.window, "提示", "请先加载模型！")
            return

        info = self.detector.get_model_info()
        if not info or "classes" not in info:
            QMessageBox.warning(self.window, "提示", "当前模型无类别信息")
            return

        class_names = info["classes"]  # dict {id: name}

        dlg = QDialog(self.window)
        dlg.setWindowTitle("类别过滤")
        dlg.setGeometry(200, 150, 360, 480)
        dlg.setStyleSheet(self.window.styleSheet())
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel("选择要显示的类别（不选=显示全部）："))

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for cid, cname in sorted(class_names.items()):
            list_widget.addItem(f"{cid}: {cname}")
        lay.addWidget(list_widget)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_reset = QPushButton("重置(显示全部)")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_reset)
        lay.addLayout(btn_row)

        def on_ok():
            selected = list_widget.selectedItems()
            if not selected:
                self._allowed_classes = None
                self.window.btn_class_filter.setText("类别过滤 (全部)")
            else:
                ids = []
                for item in selected:
                    cid = int(item.text().split(":")[0])
                    ids.append(cid)
                self._allowed_classes = ids
                self.window.btn_class_filter.setText(f"类别过滤 ({len(ids)}类)")
            dlg.close()

        def on_reset():
            self._allowed_classes = None
            self.window.btn_class_filter.setText("类别过滤 (全部)")
            dlg.close()

        btn_ok.clicked.connect(on_ok)
        btn_reset.clicked.connect(on_reset)
        dlg.exec()

    # ------------------------------------------------------------------
    #  历史记录
    # ------------------------------------------------------------------
    @Slot()
    def _on_open_history_dialog(self):
        if self._history_dialog is None or not self._history_dialog.isVisible():
            self._history_dialog = HistoryDialog(parent=self.window)
        self._history_dialog.show()
        self._history_dialog.raise_()

    def _record_detection(self, source_type: str):
        """写入一条检测历史记录"""
        model_name = os.path.basename(self.detector.current_model_path or "")
        source_text = self.window.cmb_source.currentText()
        task = self.detector.task_type or "detect"
        try:
            obj_text = self.window.lbl_objs.text()
            obj_count = int(obj_text.split(":")[-1].strip())
        except (ValueError, IndexError):
            obj_count = 0
        try:
            lat_text = self.window.lbl_latency.text()
            latency = float(lat_text.split(":")[-1].strip().replace("ms", ""))
        except (ValueError, IndexError):
            latency = 0.0
        history.add_record(
            model=model_name,
            source=source_text,
            source_type=source_type,
            obj_count=obj_count,
            latency_ms=latency,
            task=task,
        )
        # 刷新历史对话框
        if self._history_dialog and self._history_dialog.isVisible():
            self._history_dialog._load_data()

    # ------------------------------------------------------------------
    #  清理 & 运行
    # ------------------------------------------------------------------
    def _cleanup(self):
        self._stop_current_thread()
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait(5000)
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.wait(5000)
        self.window.log("程序已退出")

    def exec_app(self):
        sys.exit(self.app.exec())


# ======================================================================
if __name__ == "__main__":
    controller = YOLOAppController()
    controller.exec_app()
