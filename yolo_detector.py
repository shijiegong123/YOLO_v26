import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import BaseYOLOLoader, get_builtin_models, get_model_task, model_exists_locally


# ============================================================
# 官方内置数据集列表
# ============================================================
OFFICIAL_DATASETS = {
    "COCO (检测 80类)":     {"data": "coco.yaml",     "task": "detect",  "desc": "COCO 2017 目标检测"},
    "COCO128 (检测 80类)":  {"data": "coco128.yaml",  "task": "detect",  "desc": "COCO 子集，快速验证"},
    "COCO8 (检测 8类)":     {"data": "coco8.yaml",    "task": "detect",  "desc": "COCO 超小子集，调试用"},
    "VOC (检测 20类)":      {"data": "VOC.yaml",      "task": "detect",  "desc": "Pascal VOC 2012"},
    "Objects365 (检测)":    {"data": "Objects365.yaml","task": "detect", "desc": "Objects365 大规模检测"},
    "OpenImagesV7 (检测)":  {"data": "openImagesV7.yaml","task":"detect","desc": "Open Images V7"},
    "COCO-seg (分割)":      {"data": "coco.yaml",     "task": "segment", "desc": "COCO 实例分割"},
    "COCO8-seg (分割)":     {"data": "coco8-seg.yaml","task": "segment", "desc": "COCO 分割子集"},
    "COCO-pose (姿态)":     {"data": "coco-pose.yaml","task": "pose",    "desc": "COCO 姿态估计"},
    "COCO8-pose (姿态)":    {"data": "coco8-pose.yaml","task": "pose",   "desc": "COCO 姿态子集"},
    "DOTA-v2 (OBB)":        {"data": "DOTAv2.yaml",   "task": "obb",     "desc": "DOTA 旋转框"},
    "DOTAv1.5 (OBB)":       {"data": "DOTAv1.5.yaml", "task": "obb",     "desc": "DOTA v1.5 旋转框"},
    "Imagenet (分类)":      {"data": "imagenet.yaml", "task": "classify","desc": "ImageNet 1000类"},
}

# 支持的导出格式
EXPORT_FORMATS = {
    "PyTorch (.pt)":      {"format": "torchscript",  "ext": ".torchscript",  "desc": "TorchScript 格式"},
    "ONNX (.onnx)":       {"format": "onnx",          "ext": ".onnx",        "desc": "跨平台通用格式"},
    "TensorRT (.engine)": {"format": "engine",        "ext": ".engine",      "desc": "NVIDIA GPU 推理加速"},
    "CoreML (.mlpackage)":{"format": "coreml",        "ext": ".mlpackage",   "desc": "Apple 设备专用"},
    "OpenVINO (.xml)":    {"format": "openvino",      "ext": "_openvino_model/", "desc": "Intel 推理加速"},
    "TFLite (.tflite)":   {"format": "tflite",        "ext": ".tflite",      "desc": "移动端/嵌入式"},
    "TF.js":              {"format": "tfjs",           "ext": "_web_model/",  "desc": "浏览器推理"},
    "PaddlePaddle":       {"format": "paddle",        "ext": "_paddle_model/","desc": "PaddlePaddle 格式"},
    "NCNN (.param)":      {"format": "ncnn",          "ext": "_ncnn_model/", "desc": "腾讯 NCNN 移动端"},
}


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

    # ------------------------------------------------------------------
    #  模型训练
    # ------------------------------------------------------------------
    @staticmethod
    def get_official_datasets():
        """返回官方数据集列表。"""
        return OFFICIAL_DATASETS

    @staticmethod
    def get_export_formats():
        """返回支持的导出格式列表。"""
        return EXPORT_FORMATS

    def train(self, data: str, epochs: int = 100, imgsz: int = 640,
              batch: int = 16, device: str = None, project: str = "runs/train",
              name: str = "yolo_train", pretrained: str = None,
              task: str = "detect", extra_args: dict = None,
              on_epoch_callback=None, on_log_callback=None):
        """
        训练 YOLO 模型。
        :param data: 数据集 YAML 文件路径或官方数据集名 (如 coco128.yaml)
        :param epochs: 训练轮数
        :param imgsz: 输入图像尺寸
        :param batch: 批大小 (-1 为自动)
        :param device: 训练设备 (None=自动选择)
        :param project: 输出项目目录
        :param name: 实验名称
        :param pretrained: 预训练模型路径 (None=使用当前已加载模型)
        :param task: 任务类型
        :param extra_args: 额外训练参数字典
        :param on_epoch_callback: 每轮结束回调 fn(epoch, total_epochs, metrics_dict)
        :param on_log_callback: 日志回调 fn(message_str)
        :return: 训练后的模型路径 (best.pt)
        """
        # 准备模型
        if pretrained:
            model = YOLO(pretrained)
        elif self.is_loaded():
            model = self.model
        else:
            raise RuntimeError("未加载模型，请指定预训练模型或先加载模型")

        # 构建训练参数
        train_args = {
            "data": data,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device or self.device,
            "project": project,
            "name": name,
            "task": task,
            "verbose": True,
            "exist_ok": True,
            "plots": True,
            "save": True,
        }
        if extra_args:
            train_args.update(extra_args)

        if on_log_callback:
            on_log_callback(f"开始训练 — 数据集: {data}, Epochs: {epochs}, "
                            f"ImgSz: {imgsz}, Batch: {batch}, Device: {train_args['device']}")

        # 执行训练
        results = model.train(**train_args)

        # 获取最佳模型路径
        best_model_path = str(model.trainer.best)
        if on_log_callback:
            on_log_callback(f"训练完成！最佳模型保存至: {best_model_path}")

        return best_model_path

    def export_model(self, model_path: str = None, export_format: str = "onnx",
                     imgsz: int = 640, half: bool = False,
                     simplify: bool = True, opset: int = 12,
                     dynamic: bool = False, on_log_callback=None):
        """
        导出模型到指定格式。
        :param model_path: 模型路径 (None=使用当前加载的模型)
        :param export_format: 导出格式 (onnx/engine/torchscript/coreml/tflite 等)
        :param imgsz: 导出图像尺寸
        :param half: 是否使用 FP16 半精度
        :param simplify: 是否简化 ONNX 模型
        :param opset: ONNX opset 版本
        :param dynamic: 是否使用动态输入尺寸
        :param on_log_callback: 日志回调
        :return: 导出文件路径
        """
        if model_path:
            model = YOLO(model_path)
        elif self.is_loaded():
            model = self.model
        else:
            raise RuntimeError("未加载模型，请指定模型路径或先加载模型")

        export_args = {
            "format": export_format,
            "imgsz": imgsz,
            "half": half,
            "simplify": simplify,
            "opset": opset,
            "dynamic": dynamic,
            "device": self.device,
        }

        if on_log_callback:
            on_log_callback(f"开始导出模型 — 格式: {export_format}, 尺寸: {imgsz}")

        exported_path = model.export(**export_args)

        if on_log_callback:
            on_log_callback(f"模型导出完成: {exported_path}")

        return str(exported_path)
