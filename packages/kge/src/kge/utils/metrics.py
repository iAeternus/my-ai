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
    B, N = scores.shape

    # filtered: 将已知真三元组（非当前 target）的得分设为 -inf
    if filter_set is not None and heads is not None and relations is not None:
        mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(B):
            h, r = heads[i].item(), relations[i].item()
            for t in range(N):
                if t != true_tails[i].item() and (h, r, t) in filter_set:
                    mask[i, t] = False
        scores = scores.masked_fill(~mask, float("-inf"))

    # 排名：得分 >= true_tail 得分的实体数
    true_scores = scores[torch.arange(B), true_tails].unsqueeze(1)  # (B, 1)
    higher = (scores > true_scores).sum(dim=-1)  # (B,)
    equal = (scores == true_scores).sum(dim=-1)  # handle ties
    return higher + (equal + 1) // 2  # 1-indexed, ties get middle rank


def mrr(ranks_tensor: Tensor) -> float:
    """Mean Reciprocal Rank

    Args:
        ranks_tensor: (B,) 排名张量 (1-indexed)
    """
    return (1.0 / ranks_tensor.float()).mean().item()


def hits_at_k(ranks_tensor: Tensor, ks: list[int] = [1, 3, 10]) -> dict[int, float]:
    """Hits@K

    Args:
        ranks_tensor: (B,) 排名张量 (1-indexed)
        ks: K 值列表
    """
    total = ranks_tensor.numel()
    return {k: (ranks_tensor <= k).sum().item() / total for k in ks}


def accuracy(output: Tensor, target: Tensor) -> float:
    """多分类准确率"""
    pred = output.argmax(dim=-1)
    return (pred == target).float().mean().item()
