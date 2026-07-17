"""图结构数据集抽象"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from core.datasets.base import BaseDataset, ensure_loaded


class GraphDataset(BaseDataset):
    """图结构数据集 ABC"""

    @property
    @abstractmethod
    @ensure_loaded
    def num_nodes(self) -> int:
        """图中节点总数"""
        ...

    @property
    @abstractmethod
    @ensure_loaded
    def num_edges(self) -> int:
        """图中边总数"""
        ...

    @property
    @abstractmethod
    @ensure_loaded
    def num_features(self) -> int:
        """节点特征维度"""
        ...

    @property
    @abstractmethod
    @ensure_loaded
    def num_classes(self) -> int:
        """分类任务的类别数"""
        ...

    @property
    @ensure_loaded
    def edge_index(self) -> Tensor | None:
        """边索引 (2, E)，默认返回 None（子类按需覆盖）。"""
        return None
