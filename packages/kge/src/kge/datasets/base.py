"""知识图谱数据集基类。

继承自 ``core.BaseDataset``，复用其 lazy-loading 机制。
子类需实现 ``_load()``，在其中加载三元组数据。
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from core.datasets.base import BaseDataset, ensure_loaded  # [shared] 统一从 core 导入


class BaseKGDataset(BaseDataset):
    """知识图谱数据集基类。

    继承自 ``core.BaseDataset``（已包含 ``torch.utils.data.Dataset`` + ``ABC``），
    复用 ``root``、``_loaded`` 属性和 ``ensure_loaded`` 延迟加载机制。

    子类需实现 ``num_entities``、``num_relations``，
    并在 ``_load()`` 中填充 ``_train/_val/_test_triples``。
    """

    name: str = ""

    def __init__(self, root: str | Path) -> None:
        # [shared] super().__init__() 设置 self.root 和 self._loaded
        super().__init__(root)

        # 占位张量：_load() 中会被替换为实际数据
        # 初始化为空张量以满足类型检查器（替代原先无赋值的类型注解）
        self._train_triples: Tensor = torch.empty(0, 3, dtype=torch.long)
        self._val_triples: Tensor = torch.empty(0, 3, dtype=torch.long)
        self._test_triples: Tensor = torch.empty(0, 3, dtype=torch.long)

        self._entity_to_id: dict[str, int] = {}
        self._relation_to_id: dict[str, int] = {}
        self._id_to_entity: dict[int, str] = {}
        self._id_to_relation: dict[int, str] = {}

    # ── 抽象属性 ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def num_entities(self) -> int: ...

    @property
    @abstractmethod
    def num_relations(self) -> int: ...

    # ── PyTorch Dataset 接口 ────────────────────────────────────────

    @ensure_loaded
    def __len__(self) -> int:
        return len(self._train_triples)

    @ensure_loaded
    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        h, r, t = self._train_triples[idx]
        return int(h), int(r), int(t)

    # ── 子类实现 ────────────────────────────────────────────────────

    @abstractmethod
    def _load(self) -> None:
        """子类实现：加载数据到 self._train/_val/_test_triples"""
        ...

    # ── 三元组属性（延迟加载）───────────────────────────────────────

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
        """返回 train+val+test 的全部三元组（用于构建 filter set）。"""
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

    # ── 工具方法 ────────────────────────────────────────────────────

    @staticmethod
    def _read_triples(
        path: Path,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> list[tuple[int, int, int]]:
        """读取三元组文件，返回 (h_id, r_id, t_id) 列表。"""
        triples: list[tuple[int, int, int]] = []
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
                triples.append(
                    (entity_to_id[h], relation_to_id[r], entity_to_id[t])
                )
        return triples
