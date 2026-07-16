"""KGE 损失函数 — 纯函数，无状态"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def margin_ranking_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """Margin Ranking Loss: L = mean(relu(margin - pos + neg))

    Args:
        pos_scores: (B,) 正例得分
        neg_scores: (B, K) 负例得分
        margin: 边际值
    """
    pos_expanded = pos_scores.unsqueeze(-1).expand_as(neg_scores)
    return F.relu(margin - pos_expanded + neg_scores).mean()


def adversarial_margin_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    """自对抗 Margin Ranking Loss（RotatE）

    负例按 softmax(neg_scores / temperature) 权重加权

    Args:
        pos_scores: (B,) 正例得分
        neg_scores: (B, K) 负例得分
        margin: 边际值
        temperature: 自对抗温度
    """
    pos_expanded = pos_scores.unsqueeze(-1).expand_as(neg_scores)
    raw_loss = F.relu(margin - pos_expanded + neg_scores)

    with torch.no_grad():
        weights = F.softmax(neg_scores / temperature, dim=-1)

    return (weights * raw_loss).sum(dim=-1).mean()


def bce_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Binary Cross Entropy Loss（含标签平滑）

    Args:
        pos_scores: (B,) 正例得分
        neg_scores: (B, K) 负例得分
        label_smoothing: 标签平滑
    """
    B, K = neg_scores.shape
    pos_labels = torch.ones_like(pos_scores) * (1.0 - label_smoothing)
    neg_labels = torch.zeros_like(neg_scores) + label_smoothing / K

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, pos_labels, reduction="sum"
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, neg_labels, reduction="sum"
    )

    return (pos_loss + neg_loss) / (B * (1 + K))


def cross_entropy_1n(
    scores: Tensor,
    targets: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """1-N 交叉熵损失（ConvE 风格）

    Args:
        scores: (B, num_entities) 所有候选尾实体的得分
        targets: (B,) 正确尾实体 ID
        label_smoothing: 标签平滑
    """
    return F.cross_entropy(scores, targets, label_smoothing=label_smoothing)
