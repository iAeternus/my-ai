"""可复现性工具"""
from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """设置 Python、NumPy、PyTorch (CPU + CUDA) 的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
