import time
import os
import glob
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO


# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _filter_results(result, allowed_classes: list | None):
    """
    根据允许的类别列表过滤检测结果。
    若 allowed_classes 为 None 或空，则不过滤。
    """
    if not allowed_classes or result.boxes is None:
        return result
    cls_ids = result.boxes.cls.int().tolist()
    keep = [i for i, c in enumerate(cls_ids) if c in allowed_classes]
    if len(keep) == len(cls_ids):
        return result  # 无变化
    result.boxes = result.boxes[keep]
    return result


class DetectionThread(QThread):
    """
    视频/摄像头实时检测工作线程。
    在独立线程中运行推理循环，通过信号将处理后的帧和状态信息发送回主线程。
    """

    sig_frame_processed = Signal(object)          # 处理后的帧 (numpy array)
    sig_status_update = Signal(float, int, float) # (fps, 目标数量, 推理耗时ms)
    sig_error = Signal(str)                       # 错误消息
    sig_detection_done = Signal()                 # 检测循环结束

    def __init__(self, source, model_path: str, task_type: str = "detect",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = "cpu", save_dir: str = "",
                 allowed_classes: list | None = None):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.task_type = task_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.save_dir = save_dir
        self.allowed_classes = allowed_classes

        self._is_running = True
        self.cap = None
        self.model = None

    def run(self):
        """线程主体：加载模型 → 打开视频源 → 逐帧推理 → 发送信号"""
        self._is_running = True

        # 1) 在线程中加载模型（避免主线程阻塞和跨线程问题）
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.sig_error.emit(f"模型加载失败: {e}")
            self.sig_detection_done.emit()
            return

        # 2) 打开视频源
        try:
            self.cap = cv2.VideoCapture(self.source)
        except Exception as e:
            self.sig_error.emit(f"无法打开视频源: {e}")
            self.sig_detection_done.emit()
            return

        if not self.cap.isOpened():
            self.sig_error.emit(f"无法打开视频源: {self.source}")
            self.sig_detection_done.emit()
            return

        # 3) 逐帧推理循环
        fps_history: list[float] = []
        use_half = (self.device == "cuda")

        while self._is_running:
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频结束或摄像头断开

            t0 = time.perf_counter()
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device=self.device,
                    imgsz=640,
                    half=use_half,
                    stream=False,
                )
                result = results[0]
                result = _filter_results(result, self.allowed_classes)
                processed_frame = result.plot()

                # 统计目标数量
                obj_count = 0
                if result.boxes is not None:
                    obj_count = len(result.boxes)

                # 保存结果
                if self.save_dir:
                    os.makedirs(self.save_dir, exist_ok=True)
                    frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    save_path = os.path.join(self.save_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(save_path, processed_frame)

            except Exception as e:
                self.sig_error.emit(f"推理错误: {e}")
                continue

            # 4) 计算 FPS（移动平均，窗口 = 30 帧）
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            current_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 5) 发送信号
            self.sig_frame_processed.emit(processed_frame)
            self.sig_status_update.emit(avg_fps, obj_count, elapsed_ms)

        # 循环结束
        self._release_resources()
        self.sig_detection_done.emit()

    def stop(self):
        """请求线程停止（线程安全）"""
        self._is_running = False

    def _release_resources(self):
        """释放视频捕获和模型资源"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.model = None


class ImageDetectionThread(QThread):
    """
    单张图片检测工作线程。
    在独立线程中执行单次推理，完成后通过信号返回结果图像。
    """

    sig_image_ready = Signal(object)  # 处理后的图像 (numpy array)
    sig_status_update = Signal(float, int, float)  # (fps占位0, 目标数, 耗时ms)
    sig_error = Signal(str)
    sig_done = Signal()

    def __init__(self, image_path: str, model_path: str, task_type: str = "detect",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = "cpu", save_dir: str = "",
                 allowed_classes: list | None = None):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.task_type = task_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.save_dir = save_dir
        self.allowed_classes = allowed_classes

    def run(self):
        """线程主体：加载模型 → 推理单张图片 → 发送结果"""
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.sig_error.emit(f"模型加载失败: {e}")
            self.sig_done.emit()
            return

        t0 = time.perf_counter()
        try:
            results = model.predict(
                source=self.image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                imgsz=640,
                half=(self.device == "cuda"),
            )
            result = results[0]
            result = _filter_results(result, self.allowed_classes)
            processed = result.plot()

            obj_count = 0
            if result.boxes is not None:
                obj_count = len(result.boxes)

            # 保存结果
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                base = os.path.basename(self.image_path)
                save_path = os.path.join(self.save_dir, f"det_{base}")
                cv2.imwrite(save_path, processed)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.sig_image_ready.emit(processed)
            self.sig_status_update.emit(0.0, obj_count, elapsed_ms)

        except Exception as e:
            self.sig_error.emit(f"图片检测失败: {e}")

        self.sig_done.emit()


class TrainingThread(QThread):
    """
    模型训练工作线程。
    在独立线程中执行 YOLO 训练，通过信号实时反馈训练进度。
    """

    sig_log = Signal(str)                         # 训练日志消息
    sig_epoch_done = Signal(int, int)             # (当前轮, 总轮数)
    sig_progress = Signal(float)                  # 训练进度百分比 0~100
    sig_finished = Signal(str)                    # 训练完成，发送 best.pt 路径
    sig_error = Signal(str)                       # 错误消息

    def __init__(self, pretrained_model: str, data: str, epochs: int = 100,
                 imgsz: int = 640, batch: int = 16, device: str = "cpu",
                 project: str = "runs/train", name: str = "yolo_train",
                 task: str = "detect", extra_args: dict = None):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project = project
        self.name = name
        self.task = task
        self.extra_args = extra_args or {}
        self._is_running = True

    def run(self):
        """线程主体：加载模型 → 执行训练 → 发送结果"""
        # 任务对应的从头训练 YAML 配置
        _TASK_YAML = {
            "detect":   "yolo11n.yaml",
            "segment":  "yolo11n-seg.yaml",
            "pose":     "yolo11n-pose.yaml",
            "obb":      "yolo11n-obb.yaml",
            "classify": "yolo11n-cls.yaml",
        }
        try:
            if self.pretrained_model:
                self.sig_log.emit(f"正在加载预训练模型: {self.pretrained_model}")
                model = YOLO(self.pretrained_model)
            else:
                # 从头训练：使用模型架构 YAML 创建随机初始化模型
                yaml_config = _TASK_YAML.get(self.task, "yolo11n.yaml")
                self.sig_log.emit(f"从头开始训练，使用配置: {yaml_config}")
                model = YOLO(yaml_config)
        except Exception as e:
            self.sig_error.emit(f"模型初始化失败: {e}")
            return

        if not self._is_running:
            return

        # 构建训练参数
        train_args = {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "project": self.project,
            "name": self.name,
            "task": self.task,
            "verbose": True,
            "exist_ok": True,
            "plots": True,
            "save": True,
        }
        train_args.update(self.extra_args)

        self.sig_log.emit(
            f"开始训练 — 数据集: {self.data}, Epochs: {self.epochs}, "
            f"ImgSz: {self.imgsz}, Batch: {self.batch}, Device: {self.device}"
        )

        try:
            current_epoch = [0]

            def epoch_callback(trainer):
                if not self._is_running:
                    raise KeyboardInterrupt("用户取消训练")
                current_epoch[0] += 1
                ep = current_epoch[0]
                total = self.epochs
                self.sig_epoch_done.emit(ep, total)
                self.sig_progress.emit(ep / total * 100.0)

                try:
                    self.sig_log.emit(f"Epoch {ep}/{total} 完成")
                except Exception:
                    self.sig_log.emit(f"Epoch {ep}/{total} 完成")

            # 注册回调
            model.add_callback("on_train_epoch_end", epoch_callback)

            # 执行训练
            model.train(**train_args)

            if self._is_running:
                best_path = str(model.trainer.best)
                self.sig_progress.emit(100.0)
                self.sig_log.emit(f"训练完成！最佳模型: {best_path}")
                self.sig_finished.emit(best_path)
            else:
                self.sig_log.emit("训练已被用户取消")

        except KeyboardInterrupt:
            self.sig_log.emit("训练已被用户取消")
        except Exception as e:
            self.sig_error.emit(f"训练失败: {e}")
            import traceback
            self.sig_log.emit(traceback.format_exc())

    def stop(self):
        """请求停止训练"""
        self._is_running = False


class ExportThread(QThread):
    """
    模型导出工作线程。
    在独立线程中执行模型格式转换。
    """

    sig_log = Signal(str)
    sig_finished = Signal(str)    # 导出文件路径
    sig_error = Signal(str)
    sig_done = Signal()

    def __init__(self, model_path: str, export_format: str = "onnx",
                 imgsz: int = 640, half: bool = False,
                 simplify: bool = True, opset: int = 12,
                 dynamic: bool = False, device: str = "cpu"):
        super().__init__()
        self.model_path = model_path
        self.export_format = export_format
        self.imgsz = imgsz
        self.half = half
        self.simplify = simplify
        self.opset = opset
        self.dynamic = dynamic
        self.device = device

    def run(self):
        try:
            self.sig_log.emit(f"正在加载模型: {self.model_path}")
            model = YOLO(self.model_path)

            self.sig_log.emit(
                f"开始导出 — 格式: {self.export_format}, "
                f"ImgSz: {self.imgsz}, Half: {self.half}"
            )

            export_args = {
                "format": self.export_format,
                "imgsz": self.imgsz,
                "half": self.half,
                "simplify": self.simplify,
                "opset": self.opset,
                "dynamic": self.dynamic,
                "device": self.device,
            }

            exported_path = model.export(**export_args)
            exported_path = str(exported_path)

            self.sig_log.emit(f"导出完成: {exported_path}")
            self.sig_finished.emit(exported_path)

        except Exception as e:
            self.sig_error.emit(f"导出失败: {e}")
            import traceback
            self.sig_log.emit(traceback.format_exc())

        self.sig_done.emit()


class BatchImageThread(QThread):
    """
    批量图片检测工作线程。
    遍历指定文件夹中的所有图片，逐张推理并通过信号发送结果。
    """

    sig_image_ready = Signal(object)              # 处理后的图像
    sig_status_update = Signal(float, int, float) # (进度百分比, 目标数, 耗时ms)
    sig_log = Signal(str)                         # 日志消息
    sig_error = Signal(str)                       # 错误消息
    sig_done = Signal()                           # 全部完成

    def __init__(self, folder_path: str, model_path: str, task_type: str = "detect",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = "cpu", save_dir: str = "",
                 allowed_classes: list | None = None):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.task_type = task_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.save_dir = save_dir
        self.allowed_classes = allowed_classes

    def run(self):
        # 收集图片文件
        image_files = []
        for f in sorted(os.listdir(self.folder_path)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(self.folder_path, f))

        if not image_files:
            self.sig_error.emit(f"文件夹中未找到图片: {self.folder_path}")
            self.sig_done.emit()
            return

        self.sig_log.emit(f"找到 {len(image_files)} 张图片，开始批量检测...")

        # 加载模型
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.sig_error.emit(f"模型加载失败: {e}")
            self.sig_done.emit()
            return

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        total = len(image_files)
        total_objects = 0

        for idx, img_path in enumerate(image_files):
            t0 = time.perf_counter()
            try:
                results = model.predict(
                    source=img_path,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device=self.device,
                    imgsz=640,
                    half=(self.device == "cuda"),
                )
                result = results[0]
                result = _filter_results(result, self.allowed_classes)
                processed = result.plot()

                obj_count = 0
                if result.boxes is not None:
                    obj_count = len(result.boxes)
                total_objects += obj_count

                # 保存结果
                if self.save_dir:
                    base = os.path.basename(img_path)
                    save_path = os.path.join(self.save_dir, f"det_{base}")
                    cv2.imwrite(save_path, processed)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                progress = (idx + 1) / total * 100.0

                self.sig_image_ready.emit(processed)
                self.sig_status_update.emit(progress, obj_count, elapsed_ms)

            except Exception as e:
                self.sig_error.emit(f"处理 {os.path.basename(img_path)} 失败: {e}")

        self.sig_log.emit(f"批量检测完成: {total} 张图片, 共检测 {total_objects} 个目标")
        self.sig_done.emit()
