"""数据集注册器"""

from core.utils.registry import Registry
from core.datasets.base import BaseDataset

DATASET_REGISTRY = Registry[type[BaseDataset]]("dataset", base_class=BaseDataset)
