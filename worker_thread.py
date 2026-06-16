import time
import cv2
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO


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
                 device: str = "cpu"):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.task_type = task_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

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
                processed_frame = result.plot()

                # 统计目标数量
                obj_count = 0
                if result.boxes is not None:
                    obj_count = len(result.boxes)

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
                 device: str = "cpu"):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.task_type = task_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

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
            processed = result.plot()

            obj_count = 0
            if result.boxes is not None:
                obj_count = len(result.boxes)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.sig_image_ready.emit(processed)
            self.sig_status_update.emit(0.0, obj_count, elapsed_ms)

        except Exception as e:
            self.sig_error.emit(f"图片检测失败: {e}")

        self.sig_done.emit()
