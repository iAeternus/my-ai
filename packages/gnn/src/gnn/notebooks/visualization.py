"""Notebook 可视化工具 —— 论文主题分类 & 引用预测

所有函数均返回 ``matplotlib.figure.Figure``，调用方可用 ``plt.show()`` 展示
或 ``fig.savefig(...)`` 保存。
"""

from __future__ import annotations

from typing import Sequence

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import Data

# CJK 字体
_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
]


def _configure_cjk() -> None:
    from matplotlib.font_manager import FontProperties, fontManager

    for name in _CJK_FONT_CANDIDATES:
        try:
            font = FontProperties(family=name)
            path = fontManager.findfont(font, fallback_to_default=False)
            if path:
                matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams[
                    "font.sans-serif"
                ]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue


_configure_cjk()


def plot_training_curves(
    history: dict[str, list[float]],
    *,
    metric_key: str = "acc",
    metric_label: str = "Accuracy",
) -> Figure:
    """训练曲线双子图：Loss（左）+ 性能指标（右）

    Args:
        history: 训练历史，至少包含 ``loss``, ``val_loss``,
                以及 ``{metric_key}``, ``val_{metric_key}``
        metric_key: 性能指标键名（不含 val_ 前缀），如 ``"acc"`` 或 ``"auc"``
        metric_label: 右图 y 轴标签

    Returns:
        含两个子图的 Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    #  Loss
    if "loss" in history:
        ax1.plot(epochs, history["loss"], label="Train Loss", linewidth=1.2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.2)
        # 标注 best val_loss
        best_idx = int(np.argmin(history["val_loss"]))
        ax1.axvline(
            x=best_idx + 1,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Best (epoch {best_idx + 1})",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss 曲线")

    # Metric
    train_key = metric_key
    val_key = f"val_{metric_key}"
    if train_key in history:
        ax2.plot(epochs, history[train_key], label=f"Train {metric_key}", linewidth=1.2)
    if val_key in history:
        ax2.plot(epochs, history[val_key], label=f"Val {metric_key}", linewidth=1.2)
        best_idx = int(np.argmax(history[val_key]))
        best_val = history[val_key][best_idx]
        ax2.axvline(x=best_idx + 1, color="gray", linestyle="--", alpha=0.5)
        ax2.annotate(
            f"{best_val:.4f}",
            xy=(best_idx + 1, best_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="gray",
        )
        # Train-Val gap 阴影
        if train_key in history:
            train_vals = np.array(history[train_key])
            val_vals = np.array(history[val_key])
            min_len = min(len(train_vals), len(val_vals))
            ax2.fill_between(
                epochs[:min_len],
                train_vals[:min_len],
                val_vals[:min_len],
                alpha=0.08,
                color="red",
                label="Train-Val Gap",
            )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_label)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{metric_label} 曲线")

    fig.tight_layout()
    return fig


def plot_class_distribution(data: Data) -> Figure:
    """类别分布柱状图（含 train/val/test 拆分）"""
    y = data.y.cpu().numpy().ravel()
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.2), 4))
    x = np.arange(n_classes)
    width = 0.25

    for mask_tensor, label, color in [
        (data.train_mask, "Train", "#2ecc71"),
        (data.val_mask, "Val", "#3498db"),
        (data.test_mask, "Test", "#e74c3c"),
    ]:
        mask = mask_tensor.cpu().numpy().ravel()
        mask_counts = [int(np.sum((y == c) & mask)) for c in classes]
        ax.bar(
            x + width * ["Train", "Val", "Test"].index(label),
            mask_counts,
            width,
            label=label,
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylabel("节点数量")
    ax.set_title("各类别节点分布（按数据集划分）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: Sequence[str] | None = None,
    *,
    normalize: bool = True,
) -> Figure:
    """混淆矩阵热力图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名列表
        normalize: True = 行归一化（召回率），False = 绝对计数
    """
    from sklearn.metrics import confusion_matrix as sklearn_cm

    cm = sklearn_cm(y_true, y_pred)
    n = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]

    if normalize:
        cm_display = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums
        fmt = ".2f"
        title = "混淆矩阵（行归一化 -> 召回率）"
    else:
        cm_display = cm
        fmt = "d"
        title = "混淆矩阵（绝对计数）"

    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 0.9)))
    im = ax.imshow(cm_display, cmap="Blues", aspect="auto")

    for i in range(n):
        for j in range(n):
            text = f"{cm_display[i, j]:{fmt}}" if normalize else f"{cm[i, j]}"
            color = (
                "white"
                if cm_display[i, j] > (0.7 if normalize else cm.max() * 0.6)
                else "black"
            )
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


def plot_tsne_embeddings(
    embeddings: Tensor | NDArray[np.floating],
    labels: Tensor | NDArray[np.integer],
    *,
    train_mask: NDArray[np.bool_] | None = None,
    val_mask: NDArray[np.bool_] | None = None,
    test_mask: NDArray[np.bool_] | None = None,
    title: str = "t-SNE 节点嵌入可视化",
    color_map: str = "tab10",
) -> Figure:
    """t-SNE 降维 + 按类别着色的散点图

    Args:
        embeddings: 节点嵌入 [N, D]
        labels: 节点标签 [N]
        train_mask / val_mask / test_mask: 布尔 mask，不同子集用不同 marker
        title: 图标题
        color_map: matplotlib colormap 名称
    """
    from sklearn.manifold import TSNE

    if isinstance(embeddings, Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = np.unique(labels)
    cmap = plt.colormaps[color_map]

    any_mask = any(m is not None for m in (train_mask, val_mask, test_mask))
    markers = {"train": "o", "val": "s", "test": "^"}
    masks: dict[str, NDArray[np.bool_] | None] = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    for label_idx in unique_labels:
        if any_mask:
            for split_name, mask in masks.items():
                if mask is None:
                    continue
                subset = (labels == label_idx) & mask
                if not subset.any():
                    continue
                ax.scatter(
                    reduced[subset, 0],
                    reduced[subset, 1],
                    c=[cmap(label_idx / max(1, len(unique_labels) - 1))],
                    marker=markers[split_name],
                    s=12,
                    alpha=0.6,
                    edgecolors="none",
                    label=(
                        f"Class {label_idx} ({split_name})"
                        if split_name == "train"
                        else ""
                    ),
                )
        else:
            subset = labels == label_idx
            if not subset.any():
                continue
            ax.scatter(
                reduced[subset, 0],
                reduced[subset, 1],
                c=[cmap(label_idx / max(1, len(unique_labels) - 1))],
                marker="o",
                s=10,
                alpha=0.5,
                edgecolors="none",
            )

    if any_mask:
        # 精简类别图例（每类只显示 train marker）
        handles, lbls = ax.get_legend_handles_labels()
        by_label: dict[str, object] = {}
        for h, l in zip(handles, lbls):
            base = l.rsplit(" (", 1)[0]
            by_label.setdefault(base, h)
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7,
            markerscale=1.5,
            loc="upper right",
        )

        # 自定义 marker 图例
        from matplotlib.lines import Line2D

        marker_legend = [
            Line2D([0], [0], marker="o", color="gray", linestyle="none", label="Train"),
            Line2D([0], [0], marker="s", color="gray", linestyle="none", label="Val"),
            Line2D([0], [0], marker="^", color="gray", linestyle="none", label="Test"),
        ]
        ax.legend(handles=marker_legend, fontsize=7, loc="lower right")
    else:
        # 无 mask 时显示 colorbar 表示连续值（度数场景）
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max()),
        )
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("度数", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_edge_split_summary(
    train_data: Data,
    val_data: Data,
    test_data: Data,
) -> Figure:
    """边分割可视化 —— 正负样本分布 & 各 split 边数量

    Args:
        train_data / val_data / test_data: ``RandomLinkSplit`` 返回的三个 Data
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：各 split 边数量
    splits = []
    pos_counts = []
    neg_counts = []
    for name, d in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        lbl = d.edge_label.numpy()
        pos = int((lbl == 1).sum())
        neg = int((lbl == 0).sum())
        splits.append(name)
        pos_counts.append(pos)
        neg_counts.append(neg)

    x = np.arange(len(splits))
    w = 0.35
    axes[0].bar(
        x - w / 2, pos_counts, w, label="正样本 (有边)", color="#2ecc71", alpha=0.85
    )
    axes[0].bar(
        x + w / 2, neg_counts, w, label="负样本 (无边)", color="#e74c3c", alpha=0.85
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_ylabel("边对数量")
    axes[0].set_title("各数据集的监督边数量")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2, axis="y")

    # 右：正负样本比例饼图
    total_pos = sum(pos_counts)
    total_neg = sum(neg_counts)
    axes[1].pie(
        [total_pos, total_neg],
        labels=[f"正样本\n({total_pos})", f"负样本\n({total_neg})"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.03),
    )
    axes[1].set_title("正负样本比例")

    fig.tight_layout()
    return fig


def plot_roc_pr_curves(
    y_true: NDArray[np.floating],
    y_score: NDArray[np.floating],
) -> Figure:
    """ROC + PR 曲线双子图

    Args:
        y_true: 真实标签
        y_score: 预测分数（未经过 sigmoid）
    """
    from sklearn.metrics import (
        RocCurveDisplay,
        PrecisionRecallDisplay,
        roc_auc_score,
        average_precision_score,
    )

    y_true_b = y_true.astype(int)
    y_prob = 1 / (1 + np.exp(-y_score))  # sigmoid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ROC
    auc = roc_auc_score(y_true_b, y_prob)
    RocCurveDisplay.from_predictions(
        y_true_b, y_prob, ax=ax1, name=f"GNN (AUC={auc:.4f})"
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax1.set_title("ROC 曲线")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # PR
    ap = average_precision_score(y_true_b, y_prob)
    PrecisionRecallDisplay.from_predictions(
        y_true_b, y_prob, ax=ax2, name=f"GNN (AP={ap:.4f})"
    )
    baseline = y_true_b.mean()
    ax2.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({baseline:.3f})",
    )
    ax2.set_title("Precision-Recall 曲线")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def plot_prediction_histogram(
    y_true: NDArray[np.floating],
    y_score: NDArray[np.floating],
    *,
    bins: int = 50,
) -> Figure:
    """预测分数分布直方图（正/负样本分色）

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数（未经过 sigmoid）
        bins: 直方图柱数
    """
    y_true_b = y_true.astype(int)
    y_prob = 1 / (1 + np.exp(-y_score))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        y_prob[y_true_b == 1],
        bins=bins,
        alpha=0.6,
        color="#2ecc71",
        label=f"正样本 (n={int(y_true_b.sum())})",
        density=True,
    )
    ax.hist(
        y_prob[y_true_b == 0],
        bins=bins,
        alpha=0.6,
        color="#e74c3c",
        label=f"负样本 (n={int((1 - y_true_b).sum())})",
        density=True,
    )
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="阈值 0.5")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("密度")
    ax.set_title("预测分数分布（正负样本对比）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_top_predictions_table(
    data: Data,
    y_score: NDArray[np.floating],
    *,
    top_k: int = 10,
) -> Figure:
    """Top-K 预测推荐表格图（高分数但标签为负的论文对）

    Returns:
        matplotlib Figure（含表格）
    """
    y_prob = 1 / (1 + np.exp(-y_score))
    edge_index = data.edge_label_index.numpy()
    edge_label = data.edge_label.numpy().astype(int)

    # 筛选预测为正但实际为负的（高分误判）+ 最高分的正样本
    mistakes = np.argsort(-y_prob)
    rows = []
    for idx in mistakes[: top_k * 2]:
        if len(rows) >= top_k:
            break
        src, dst = edge_index[0, idx], edge_index[1, idx]
        rows.append(
            [
                f"#{src} ↔ #{dst}",
                f"{y_prob[idx]:.4f}",
                "✓" if edge_label[idx] == 1 else "✗",
            ]
        )

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["论文对", "预测概率", "真实标签"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # 样式
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#ddd")
        if key[0] == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")

    ax.set_title("Top-K 预测推荐（高置信度论文对）", fontweight="bold", pad=20)
    fig.tight_layout()
    return fig
