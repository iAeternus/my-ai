"""KG 数据集注册中心。

基于 ``core.Registry``，替代旧的 ``KGDatasetRegistry`` 类。
"""

from core import Registry
from kge.datasets.base import BaseKGDataset

KG_DATASET_REGISTRY = Registry[type[BaseKGDataset]](
    "kg dataset", base_class=BaseKGDataset
)
