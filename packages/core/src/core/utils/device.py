"""设备选择工具"""

from __future__ import annotations
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """解析设备偏好

    Args:
        preferred: "auto" -> CUDA > CPU; "cuda"/"cpu" -> 指定设备
    """
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)
