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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """为每个正例采样 num_neg 个负例

        Returns:
            (neg_heads, neg_relations, neg_tails): 每个 shape (B * num_neg,)
        """
        ...

    def _resample_collision(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        neg_h: Tensor,
        neg_t: Tensor,
        device: torch.device,
        filter_set: set | None = None,
        max_attempts: int = 20,
    ) -> tuple[Tensor, Tensor]:
        """碰撞重采样，替换与正例或 filter_set 冲突的负样本"""
        B = h.size(0)
        for _ in range(max_attempts):
            mask = torch.ones(B, dtype=torch.bool)
            for i in range(B):
                nh, nr, nt = neg_h[i].item(), r[i].item(), neg_t[i].item()
                if (nh, nr, nt) == (h[i].item(), r[i].item(), t[i].item()):
                    mask[i] = False
                elif filter_set is not None and (nh, nr, nt) in filter_set:
                    mask[i] = False
            if mask.all():
                break
            num_resample = int((~mask).sum().item())
            neg_h[~mask] = torch.randint(
                0,
                self.num_entities,
                (num_resample,),
                device=device,
            )
            neg_t[~mask] = torch.randint(
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = head.size(0)
        total = B * num_neg

        neg_heads = head.repeat_interleave(num_neg)
        neg_relations = relation.repeat_interleave(num_neg)
        neg_tails = tail.repeat_interleave(num_neg)

        # 随机决定替换头还是尾
        replace_head = torch.randint(0, 2, (total,), dtype=torch.bool, device=device)
        num_replace_head, num_replace_tail = int(replace_head.sum()), int(
            ~replace_head.sum()
        )
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
        self._prob_replace_head = (tph / (tph + hpt + 1e-8)).float()

    def sample(
        self,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        num_neg: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        neg_heads = head.repeat_interleave(num_neg)
        neg_relations = relation.repeat_interleave(num_neg)
        neg_tails = tail.repeat_interleave(num_neg)

        probs = self._prob_replace_head[relation].repeat_interleave(num_neg)
        replace_head = torch.bernoulli(probs).bool()
        num_replace_head, num_replace_tail = int(replace_head.sum()), int(
            ~replace_head.sum()
        )
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

        return neg_heads, neg_relations, neg_tails
