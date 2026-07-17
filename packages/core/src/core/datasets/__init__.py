"""core.datasets — 数据集抽象层。

提供跨领域数据集的 ABC、注册器、下载/缓存工具和划分工具。
具体数据集实现（FB15k237、Cora 等）留在各包中。
"""

from core.datasets.base import BaseDataset, ensure_loaded
from core.datasets.cache import download_file, extract_tar, extract_zip
from core.datasets.graph import GraphDataset
from core.datasets.registry import DATASET_REGISTRY
from core.datasets.split import random_split_indices, stratified_split_indices
from core.datasets.triplet import TripletDataset

__all__ = [
    "BaseDataset",
    "DATASET_REGISTRY",
    "GraphDataset",
    "TripletDataset",
    "download_file",
    "ensure_loaded",
    "extract_tar",
    "extract_zip",
    "random_split_indices",
    "stratified_split_indices",
]
