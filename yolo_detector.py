import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import BaseYOLOLoader, get_builtin_models, get_model_task, model_exists_locally


class YOLO_Detector(BaseYOLOLoader):
    """
    YOLO 目标检测核心类。
    支持 Detect / Segment / Pose / OBB / Classify 五种任务类型，
    支持模型自动下载和切换。
    """

    def __init__(self):
        super().__init__()
        self.task_type = "detect"
        self._on_download_callback = None  # 可选：下载进度回调

    # ------------------------------------------------------------------
    #  模型管理
    # ------------------------------------------------------------------
    def set_download_callback(self, callback):
        """
        设置模型下载进度回调函数。
        callback(message: str) 在下载开始时被调用。
        """
        self._on_download_callback = callback

    def load_model(self, model_path: str, task_type: str = "detect"):
        """
        加载模型。
        - 如果 model_path 为文件名且本地不存在，Ultralytics 会自动从服务器下载。
        - 支持任务类型: detect / segment / pose / obb / classify
        :param model_path: 模型路径或文件名 (如 yolo26s.pt)
        :param task_type: 任务类型
        :return: 是否加载成功
        """
        try:
            # 卸载旧模型释放显存
            if self.model is not None:
                self.unload_model()

            # 若本地不存在，提示即将下载
            base_name = os.path.basename(model_path)
            if not model_exists_locally(base_name) and not os.path.isabs(model_path):
                msg = f"模型 {base_name} 本地不存在，正在从 Ultralytics 服务器下载..."
                print(msg)
                if self._on_download_callback:
                    self._on_download_callback(msg)

            print(f"正在加载模型: {model_path} ...")
            self.model = YOLO(model_path)
            self.current_model_path = model_path

            # 从模型自身推断真实任务类型
            real_task = getattr(self.model, "task", task_type)
            self.task_type = real_task

            if real_task != task_type and task_type != "detect":
                print(
                    f"[提示] 所选任务({task_type})与模型实际任务({real_task})不一致，"
                    f"已自动切换为 {real_task}"
                )

            print(f"模型加载成功 — 设备: {self.device}, 任务: {self.task_type}")
            return True

        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def list_available_models(self):
        """
        返回内置可用模型列表。
        :return: list of (显示名, 文件名, 任务类型, 描述)
        """
        return get_builtin_models()

    # ------------------------------------------------------------------
    #  图片检测
    # ------------------------------------------------------------------
    def detect_image(self, image_path: str, conf: float = 0.25, iou: float = 0.45):
        """
        单张图片推理。
        :param image_path: 图片路径
        :param conf: 置信度阈值
        :param iou: IOU 阈值
        :return: 绘制后的 BGR 图像 (numpy array), results[0]
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先加载模型")

        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self.device,
            imgsz=640,
            half=(self.device == "cuda"),
        )
        result = results[0]
        drawn = self.draw_detections(result)
        return drawn, result

    def detect_image_file(self, image_path: str, save_path: str = None,
                          conf: float = 0.25, iou: float = 0.45):
        """
        检测单张图片并可选保存结果。
        :param image_path: 输入图片路径
        :param save_path: 保存路径（None 则不保存）
        :param conf: 置信度阈值
        :param iou: IOU 阈值
        :return: 绘制后的图像, 检测结果对象
        """
        drawn, result = self.detect_image(image_path, conf, iou)

        if save_path:
            cv2.imwrite(save_path, drawn)
            print(f"检测结果已保存至: {save_path}")

        return drawn, result

    # ------------------------------------------------------------------
    #  视频帧处理
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        """
        处理单帧视频/摄像头画面。
        :param frame: OpenCV BGR 图像
        :param conf: 置信度阈值
        :param iou: IOU 阈值
        :return: 绘制后的 BGR 图像, 检测结果对象
        """
        if not self.is_loaded():
            return frame, None

        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self.device,
            imgsz=640,
            half=(self.device == "cuda"),
        )
        result = results[0]
        processed = self.draw_detections(result)
        return processed, result

    # ------------------------------------------------------------------
    #  可视化
    # ------------------------------------------------------------------
    def draw_detections(self, result):
        """
        根据任务类型绘制检测结果。
        自动适配：
          - Detect : 边界框 + 类别标签 + 置信度
          - Segment: 实例分割掩码 + 边界框
          - Pose   : 关键点 + 骨架 + 边界框
          - OBB    : 旋转边界框
          - Classify: Top-5 预测
        :param result: Ultralytics 推理结果对象 (Results)
        :return: 绘制后的 BGR 图像 (numpy array)
        """
        # result.plot() 由 Ultralytics 提供，已内置对所有任务类型的支持
        return result.plot()

    # ------------------------------------------------------------------
    #  辅助信息
    # ------------------------------------------------------------------
    def get_model_info(self):
        """获取当前模型信息字典。"""
        if not self.is_loaded():
            return None

        info = {
            "path": self.current_model_path,
            "device": self.device,
            "task": self.task_type,
        }
        if hasattr(self.model, "names"):
            info["classes"] = self.model.names
            info["num_classes"] = len(self.model.names)
        return info

    @staticmethod
    def get_task_display_name(task: str) -> str:
        """将任务标识转为中文显示名。"""
        mapping = {
            "detect": "目标检测",
            "segment": "实例分割",
            "pose": "姿态估计",
            "obb": "旋转框检测",
            "classify": "图像分类",
        }
        return mapping.get(task, task)
