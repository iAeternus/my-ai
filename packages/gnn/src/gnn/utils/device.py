import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """返回计算设备。preferred="auto" 时自动选择 CUDA/CPU"""
    if preferred == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preferred)
    logger.info("使用设备: %s", device)
    return device
