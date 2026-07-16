"""负采样器"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class NegativeSampler(ABC):
    """负采样基类"""

    def __init__(self, num_entities: int) -> None:
        self.num_entities = num_entities

    @abstractmethod
    def sample(
        self,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        num_neg: int,
        device: torch.device,
        filter_set: set | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """为每个正例采样 num_neg 个负例

        Args:
            head: (B,) 头实体 ID
            relation: (B,) 关系 ID
            tail: (B,) 尾实体 ID
            num_neg: 每个正例的负样本数
            device: 目标设备
            filter_set: 可选过滤集合，用于碰撞重采样

        Returns:
            (neg_heads, neg_relations, neg_tails): 每个 shape (B * num_neg,)
        """
        ...

    @staticmethod
    def _build_hash_tensor(
        filter_set: set[tuple[int, int, int]],
        num_entities: int,
        num_relations: int,
        device: torch.device,
    ) -> Tensor:
        """将 filter_set 转为排序的完美哈希 tensor，供 searchsorted 使用

        哈希: (h * N_rel + r) * N_ent + t
        """
        N_ent = num_entities
        N_rel = num_relations
        hashes = [(h * N_rel + r) * N_ent + t for h, r, t in filter_set]
        tensor = torch.tensor(hashes, dtype=torch.long, device=device)
        return tensor.sort().values

    def _resample_collision(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        neg_h: Tensor,
        neg_t: Tensor,
        filter_hashes: Tensor | None,
        num_relations: int,
        device: torch.device,
        max_attempts: int = 20,
    ) -> tuple[Tensor, Tensor]:
        """向量化碰撞重采样

        使用完美哈希编码三元组，通过 sorted search 检测碰撞，
        替换与正例或 filter_set 冲突的负样本

        Args:
            h, r, t: (B,) 正例三元组
            neg_h: (B * num_neg,) 负样本头实体
            neg_t: (B * num_neg,) 负样本尾实体
            filter_hashes: 排序的 filter_set 哈希 tensor，None 表示不过滤
            num_relations: 关系总数（用于哈希计算）
            device: 目标设备
            max_attempts: 最大重试次数

        Returns:
            (neg_h, neg_t): 碰撞重采样后的负样本
        """
        assert neg_h.device == device and neg_t.device == device, (
            f"Device mismatch: neg_h={neg_h.device}, neg_t={neg_t.device}, expected={device}"
        )
        assert h.device == device and t.device == device

        B_pos = h.size(0)
        B_neg = neg_h.size(0)
        num_neg = B_neg // B_pos
        N_ent = self.num_entities
        N_rel = num_relations

        # 将正例展开到与负样本相同 batch 维度
        h_exp = h.repeat_interleave(num_neg)  # (B_neg,)
        r_exp = r.repeat_interleave(num_neg)  # (B_neg,)
        t_exp = t.repeat_interleave(num_neg)  # (B_neg,)

        # 正例哈希: (B_neg,)
        pos_hash = (h_exp.long() * N_rel + r_exp.long()) * N_ent + t_exp.long()

        for _ in range(max_attempts):
            # 负样本哈希: (B_neg,)
            neg_hash = (neg_h.long() * N_rel + r_exp.long()) * N_ent + neg_t.long()

            # 碰撞检测: 与正例
            collision_mask = neg_hash == pos_hash

            # 碰撞检测: 与 filter_set (binary search, O(B * log|F|))
            if filter_hashes is not None:
                idx = torch.searchsorted(filter_hashes, neg_hash)
                in_bounds = idx < len(filter_hashes)
                found = filter_hashes[idx.clamp(max=len(filter_hashes) - 1)] == neg_hash
                collision_mask = collision_mask | (in_bounds & found)

            if not collision_mask.any():
                break

            num_resample = int(collision_mask.sum().item())
            neg_h[collision_mask] = torch.randint(
                0,
                self.num_entities,
                (num_resample,),
                device=device,
            )
            neg_t[collision_mask] = torch.randint(
                0,
                self.num_entities,
                (num_resample,),
                device=device,
            )

        return neg_h, neg_t


class UniformNegativeSampler(NegativeSampler):
    """均匀负采样: 50% 替换头实体 / 50% 替换尾实体"""

    def sample(
        self,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        num_neg: int,
        device: torch.device,
        filter_set: set | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = head.size(0)
        total = B * num_neg

        neg_heads = head.repeat_interleave(num_neg)
        neg_relations = relation.repeat_interleave(num_neg)
        neg_tails = tail.repeat_interleave(num_neg)

        # 随机决定替换头还是尾
        replace_head = torch.randint(0, 2, (total,), dtype=torch.bool, device=device)
        num_replace_head = int(replace_head.sum())
        num_replace_tail = int((~replace_head).sum())

        neg_heads[replace_head] = torch.randint(
            0,
            self.num_entities,
            (num_replace_head,),
            device=device,
        )
        neg_tails[~replace_head] = torch.randint(
            0,
            self.num_entities,
            (num_replace_tail,),
            device=device,
        )

        # 过滤碰撞
        if filter_set is not None:
            filter_hashes = self._build_hash_tensor(
                filter_set,
                self.num_entities,
                int(relation.max().item()) + 1,
                device,
            )
            neg_heads, neg_tails = self._resample_collision(
                head,
                relation,
                tail,
                neg_heads,
                neg_tails,
                filter_hashes,
                int(relation.max().item()) + 1,
                device,
            )

        return neg_heads, neg_relations, neg_tails


class BernoulliNegativeSampler(NegativeSampler):
    """伯努利负采样: 按关系基数 (tph/hpt) 决定替换概率

    Args:
        tph: (num_relations,) 每个关系的 tail-per-head 平均值
        hpt: (num_relations,) 每个关系的 head-per-tail 平均值
    """

    def __init__(self, num_entities: int, tph: Tensor, hpt: Tensor) -> None:
        super().__init__(num_entities)
        # 替换头的概率 = tph / (tph + hpt)
        self._prob_replace_head: Tensor = (tph / (tph + hpt + 1e-8)).float()

    def sample(
        self,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        num_neg: int,
        device: torch.device,
        filter_set: set | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = head.size(0)

        neg_heads = head.repeat_interleave(num_neg)
        neg_relations = relation.repeat_interleave(num_neg)
        neg_tails = tail.repeat_interleave(num_neg)

        probs = self._prob_replace_head[relation].repeat_interleave(num_neg)
        replace_head = torch.bernoulli(probs).bool()
        num_replace_head = int(replace_head.sum())
        num_replace_tail = int((~replace_head).sum())

        neg_heads[replace_head] = torch.randint(
            0,
            self.num_entities,
            (num_replace_head,),
            device=device,
        )
        neg_tails[~replace_head] = torch.randint(
            0,
            self.num_entities,
            (num_replace_tail,),
            device=device,
        )

        # 过滤碰撞
        if filter_set is not None:
            num_rel = max(
                int(relation.max().item()) + 1,
                self._prob_replace_head.size(0),
            )
            filter_hashes = self._build_hash_tensor(
                filter_set,
                self.num_entities,
                num_rel,
                device,
            )
            neg_heads, neg_tails = self._resample_collision(
                head,
                relation,
                tail,
                neg_heads,
                neg_tails,
                filter_hashes,
                num_rel,
                device,
            )

        return neg_heads, neg_relations, neg_tails


class SelfAdversarialNegativeSampler(UniformNegativeSampler):
    """自对抗负采样 (RotatE)

    采样策略与 UniformNegativeSampler 完全相同（均匀替换头/尾）
    区别在于损失函数中使用 softmax(score) 对负样本加权，降低低质量负样本的影响

    本类作为标记类，方便 KGDataModule 和 Trainer 按类型分发不同的损失计算逻辑
    实际的自对抗权重计算在 loss 函数中完成

    Reference:
        Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." ICLR 2019.
    """
