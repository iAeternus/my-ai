from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


def ensure_loaded(func):
    """确保数据集已经完成加载"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._loaded:
            self._load()
            self._loaded = True
        return func(self, *args, **kwargs)

    return wrapper


class BaseKGDataset(Dataset, ABC):
    """知识图谱数据集基类

    子类需实现 num_entities, num_relations，并加载 _train/_val/_test_triples
    """

    name: str = ""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self._train_triples: Tensor
        self._val_triples: Tensor
        self._test_triples: Tensor
        self._entity_to_id: dict[str, int] = {}
        self._relation_to_id: dict[str, int] = {}
        self._id_to_entity: dict[int, str] = {}
        self._id_to_relation: dict[int, str] = {}
        self._loaded = False

    @property
    @abstractmethod
    def num_entities(self) -> int: ...

    @property
    @abstractmethod
    def num_relations(self) -> int: ...

    @ensure_loaded
    def __len__(self) -> int:
        return len(self._train_triples)

    @ensure_loaded
    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        h, r, t = self._train_triples[idx]
        return int(h), int(r), int(t)

    @abstractmethod
    def _load(self) -> None:
        """子类实现：加载数据到 self._train/_val/_test_triples"""
        ...

    @property
    @ensure_loaded
    def train_triples(self) -> Tensor:
        return self._train_triples

    @property
    @ensure_loaded
    def val_triples(self) -> Tensor:
        return self._val_triples

    @property
    @ensure_loaded
    def test_triples(self) -> Tensor:
        return self._test_triples

    @property
    @ensure_loaded
    def all_triples(self) -> Tensor:
        """返回 train+val+test 的全部三元组 (用于 filter set)"""
        parts = [self.train_triples]
        val = self.val_triples
        if val.numel() > 0:
            parts.append(val)
        test = self.test_triples
        if test.numel() > 0:
            parts.append(test)
        return Tensor(np.concatenate([p.numpy() for p in parts], axis=0))

    @property
    @ensure_loaded
    def entity_to_id(self) -> dict[str, int]:
        return self._entity_to_id

    @property
    @ensure_loaded
    def id_to_entity(self) -> dict[int, str]:
        return self._id_to_entity

    @property
    @ensure_loaded
    def relation_to_id(self) -> dict[str, int]:
        return self._relation_to_id

    @property
    @ensure_loaded
    def get_id_to_relation(self) -> dict[int, str]:
        return self._id_to_relation

    @staticmethod
    def _read_triples(
        path: Path,
        entity_to_id: dict,
        relation_to_id: dict,
    ) -> list[tuple[int, int, int]]:
        """读取三元组文件，返回 (h_id, r_id, t_id) 列表"""
        triples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                if (
                    h not in entity_to_id
                    or t not in entity_to_id
                    or r not in relation_to_id
                ):
                    continue
                triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
        return triples
