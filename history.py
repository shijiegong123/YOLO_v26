"""
历史记录管理器
使用 JSON 文件持久化存储每次检测的元数据。
"""
import json
import os
from datetime import datetime
from typing import Optional


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_history.json")


def _load_history() -> list[dict]:
    """从 JSON 文件加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_history(records: list[dict]):
    """将历史记录写入 JSON 文件"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"保存历史记录失败: {e}")


def add_record(model: str, source: str, source_type: str,
               obj_count: int, latency_ms: float, task: str = "detect",
               extra: dict = None):
    """
    添加一条检测记录。
    :param model: 模型文件名
    :param source: 检测源描述
    :param source_type: 源类型 (camera/video/image/batch)
    :param obj_count: 检测到的目标数量
    :param latency_ms: 推理耗时（毫秒）
    :param task: 任务类型
    :param extra: 附加信息字典
    """
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "task": task,
        "source": source,
        "source_type": source_type,
        "obj_count": obj_count,
        "latency_ms": round(latency_ms, 1),
    }
    if extra:
        record.update(extra)

    records = _load_history()
    records.append(record)

    # 最多保留 5000 条记录
    if len(records) > 5000:
        records = records[-5000:]

    _save_history(records)


def get_records() -> list[dict]:
    """返回所有历史记录"""
    return _load_history()


def get_statistics() -> dict:
    """
    返回基本统计信息。
    """
    records = _load_history()
    if not records:
        return {"total": 0}

    total = len(records)
    total_objects = sum(r.get("obj_count", 0) for r in records)
    avg_latency = sum(r.get("latency_ms", 0) for r in records) / total

    # 按任务类型统计
    task_counts: dict[str, int] = {}
    for r in records:
        t = r.get("task", "detect")
        task_counts[t] = task_counts.get(t, 0) + 1

    # 按源类型统计
    source_counts: dict[str, int] = {}
    for r in records:
        s = r.get("source_type", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1

    return {
        "total": total,
        "total_objects": total_objects,
        "avg_latency_ms": round(avg_latency, 1),
        "task_counts": task_counts,
        "source_counts": source_counts,
        "first_record": records[0].get("timestamp", ""),
        "last_record": records[-1].get("timestamp", ""),
    }


def clear_history():
    """清空所有历史记录"""
    _save_history([])
