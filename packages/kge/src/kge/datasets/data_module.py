"""统一数据接口层"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

from kge.config.schema import Config
from kge.datasets.base import BaseKGDataset
from kge.datasets.sampler import (
    BernoulliNegativeSampler,
    NegativeSampler,
    SelfAdversarialNegativeSampler,
    UniformNegativeSampler,
)

if TYPE_CHECKING:
    pass


@dataclass(slots=True, frozen=True)
class KGBatch:
    """统一的三元组批次数据结构

    正例: (B,) 每个 tensor
    负例: (B * num_neg,) 每个 tensor — 每个正例重复 num_neg 次后替换头/尾
    """

    pos_h: Tensor
    pos_r: Tensor
    pos_t: Tensor
    neg_h: Tensor
    neg_r: Tensor
    neg_t: Tensor


class KGDataModule:
    """知识图谱数据模块

    职责:
        - 持有 BaseKGDataset 实例
        - 构建过滤结构 (filter_set, hr_to_tails, tr_to_heads)
        - 计算 tph/hpt 统计量
        - 管理 NegativeSampler（自动选择 Uniform/Bernoulli/SelfAdversarial）
        - 提供 collate_fn（在 DataLoader 中完成负采样）
        - 创建 train/valid/test DataLoader

    Usage:
        dataset = load_dataset("fb15k-237", "data/")
        dm = KGDataModule(cfg, dataset)
        for batch in dm.train_dataloader():
            # batch 是 KGBatch，可直接送入模型
            ...
    """

    def __init__(self, cfg: Config, dataset: BaseKGDataset) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.num_entities = dataset.num_entities
        self.num_relations = dataset.num_relations

        # 构建过滤结构
        self.filter_set: set[tuple[int, int, int]]
        self.hr_to_tails: dict[tuple[int, int], set[int]]
        self.tr_to_heads: dict[tuple[int, int], set[int]]
        self._build_filter_structures()

        # 计算每关系统计量
        self.tph: Tensor
        self.hpt: Tensor
        self._compute_statistics()

        # 选择并创建负采样器
        self.negative_sampler: NegativeSampler = self._create_sampler()

    def _build_filter_structures(self) -> None:
        """遍历 all_triples 构建 filter_set, hr_to_tails, tr_to_heads

        单次 O(N) 遍历，用于:
            - filter_set: 负采样碰撞检测
            - hr_to_tails: 过滤评估 (给定 (h,r) 已知尾实体)
            - tr_to_heads: 过滤评估 (给定 (t,r) 已知头实体)
        """
        all_triples = self.dataset.all_triples  # (N, 3) LongTensor
        self.filter_set = set()
        self.hr_to_tails = {}
        self.tr_to_heads = {}

        for h, r, t in all_triples.tolist():
            h_int, r_int, t_int = int(h), int(r), int(t)
            triple = (h_int, r_int, t_int)
            self.filter_set.add(triple)
            self.hr_to_tails.setdefault((h_int, r_int), set()).add(t_int)
            self.tr_to_heads.setdefault((t_int, r_int), set()).add(h_int)

    def _compute_statistics(self) -> None:
        """从训练集计算每个关系的 tph / hpt

        tph[r] = 关系 r 的 tail-per-head 均值 = total_triples_r / unique_heads_r
        hpt[r] = 关系 r 的 head-per-tail 均值 = total_triples_r / unique_tails_r

        用于 BernoulliNegativeSampler 决定替换头/尾的概率
        """
        train = self.dataset.train_triples  # (N_train, 3) LongTensor
        num_rel = self.num_relations
        tph = torch.zeros(num_rel)
        hpt = torch.zeros(num_rel)

        for r in range(num_rel):
            mask = train[:, 1] == r
            total = mask.sum().item()
            if total == 0:
                tph[r] = 1.0
                hpt[r] = 1.0
                continue
            unique_heads = train[mask, 0].unique().numel()
            unique_tails = train[mask, 2].unique().numel()
            tph[r] = total / max(unique_heads, 1)
            hpt[r] = total / max(unique_tails, 1)

        self.tph = tph
        self.hpt = hpt

    def _create_sampler(self) -> NegativeSampler:
        """根据配置自动选择负采样器

        优先级:
            1. 若 adversarial_temperature > 0 → SelfAdversarialNegativeSampler
            2. 若 sampler_name == "bernoulli" → BernoulliNegativeSampler (需 tph/hpt)
            3. 默认 → UniformNegativeSampler
        """
        sampler_name = getattr(self.cfg.dataset, "sampler_name", None)

        if self.cfg.train.adversarial_temperature > 0:
            return SelfAdversarialNegativeSampler(self.num_entities)

        if sampler_name == "bernoulli":
            return BernoulliNegativeSampler(
                self.num_entities,
                self.tph,
                self.hpt,
            )

        return UniformNegativeSampler(self.num_entities)

    def collate_fn(self, batch: list[tuple[Tensor, Tensor, Tensor]]) -> KGBatch:
        """DataLoader collate_fn: 堆叠正例后调用负采样器生成负例

        Args:
            batch: list of (h, r, t) 0-d tensors from TensorDataset

        Returns:
            KGBatch with positive and negative triples
        """
        pos_h, pos_r, pos_t = default_collate(batch)
        pos_h = pos_h.long()
        pos_r = pos_r.long()
        pos_t = pos_t.long()

        neg_h, neg_r, neg_t = self.negative_sampler.sample(
            pos_h,
            pos_r,
            pos_t,
            num_neg=self.cfg.dataset.num_negative_samples,
            device=pos_h.device,
            filter_set=self.filter_set,
        )

        return KGBatch(
            pos_h=pos_h,
            pos_r=pos_r,
            pos_t=pos_t,
            neg_h=neg_h,
            neg_r=neg_r,
            neg_t=neg_t,
        )

    def _make_loader(self, triples: Tensor, shuffle: bool) -> DataLoader:
        """从 (N, 3) tensor 创建 DataLoader

        使用 TensorDataset 拆分三列，配合 self.collate_fn 完成负采样
        """
        ds = TensorDataset(
            triples[:, 0].long(),
            triples[:, 1].long(),
            triples[:, 2].long(),
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=(self.cfg.dataset.num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        """训练集 DataLoader (shuffle=True, 含负采样)"""
        return self._make_loader(self.dataset.train_triples, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """验证集 DataLoader (shuffle=False)"""
        return self._make_loader(self.dataset.val_triples, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """测试集 DataLoader (shuffle=False)"""
        return self._make_loader(self.dataset.test_triples, shuffle=False)

    def __iter__(self) -> Iterator[KGBatch]:
        """便捷迭代：直接遍历训练 DataLoader"""
        return iter(self.train_dataloader())
