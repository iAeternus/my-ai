"""KGE 评估指标"""

from __future__ import annotations
import torch
from torch import Tensor


def ranks(
    scores: Tensor,
    true_tails: Tensor,
    filter_set: set[tuple[int, int, int]] | None = None,
    heads: Tensor | None = None,
    relations: Tensor | None = None,
) -> Tensor:
    """计算正确尾实体的 filtered rank (1-indexed)

    Args:
        scores: (B, num_entities) 每个 (h,r) 相对所有实体的得分
        true_tails: (B,) 正确尾实体 ID
        filter_set: 已知真三元组集合 (用于 filtered 评估)
        heads: (B,) head IDs (filter 时使用)
        relations: (B,) relation IDs (filter 时使用)

    Returns:
        (B,) LongTensor，每个样本中 true_tail 的排名 (1=最佳)
    """
    ...


def mrr(ranks_tensor: Tensor) -> float:
    """Mean Reciprocal Rank

    Args:
        ranks_tensor: (B,) 排名张量 (1-indexed)
    """
    ...


def hits_at_k(ranks_tensor: Tensor, ks: list[int] = [1, 3, 10]) -> dict[int, float]:
    """Hits@K

    Args:
        ranks_tensor: (B,) 排名张量 (1-indexed)
        ks: K 值列表
    """
    ...


def accuracy(output: Tensor, target: Tensor) -> float:
    """多分类准确率"""
    ...
