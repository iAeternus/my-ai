"""数据集划分工具"""

from __future__ import annotations

import random

from torch import Tensor


def random_split_indices(
    n: int,
    ratios: tuple[float, float, float],
    *,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """将 0..n-1 随机打乱后按比例分割为 train/val/test 索引列表。

    Args:
        n: 样本总数。
        ratios: (train_ratio, val_ratio, test_ratio) 三元组。
        seed: 随机种子。

    Returns:
        (train_indices, val_indices, test_indices) 三元组。
    """
    total = sum(ratios)
    train_r, val_r, test_r = ratios[0] / total, ratios[1] / total, ratios[2] / total

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def stratified_split_indices(
    labels: Tensor,
    ratios: tuple[float, float, float],
    *,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """按标签分层拆分为 train/val/test 索引张量。

    对每个类别分别按比例划分后合并，保证各类别在各 split 中的分布一致。

    Args:
        labels: 标签张量 (N,)。
        ratios: (train_ratio, val_ratio, test_ratio) 三元组。
        seed: 随机种子。

    Returns:
        (train_indices, val_indices, test_indices) 各自为 1-D LongTensor。
    """
    total = sum(ratios)
    train_r, val_r = ratios[0] / total, ratios[1] / total

    rng = random.Random(seed)
    train_list, val_list, test_list = [], [], []

    unique_labels = labels.unique().tolist()
    for label in unique_labels:
        label_indices = (labels == label).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(label_indices)
        n = len(label_indices)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        train_list.extend(label_indices[:train_end])
        val_list.extend(label_indices[train_end:val_end])
        test_list.extend(label_indices[val_end:])

    return (
        Tensor(train_list).long(),
        Tensor(val_list).long(),
        Tensor(test_list).long(),
    )
