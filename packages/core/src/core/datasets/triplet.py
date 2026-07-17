"""TripletDataset，三元组数据集抽象"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from core.datasets.base import BaseDataset, ensure_loaded


class TripletDataset(BaseDataset):
    """三元组数据集 ABC

    定义 KGE 所需的接口：实体/关系数量、各 split 的三元组张量、
    ID 映射表。具体实现（如 FB15k237）留在各包中
    """

    @property
    @abstractmethod
    @ensure_loaded
    def num_entities(self) -> int:
        """实体总数"""
        ...

    @property
    @abstractmethod
    @ensure_loaded
    def num_relations(self) -> int:
        """关系总数"""
        ...

    @property
    @ensure_loaded
    def train_triples(self) -> Tensor:
        """训练集三元组 (N_train, 3)"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def val_triples(self) -> Tensor:
        """验证集三元组 (N_val, 3)"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def test_triples(self) -> Tensor:
        """测试集三元组 (N_test, 3)"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def all_triples(self) -> Tensor:
        """train + val + test 的全部三元组（用于 filter set 构建）"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def entity_to_id(self) -> dict[str, int]:
        """实体名 -> ID 映射"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def relation_to_id(self) -> dict[str, int]:
        """关系名 -> ID 映射"""
        raise NotImplementedError
