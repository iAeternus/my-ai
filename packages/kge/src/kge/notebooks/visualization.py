"""Notebook 可视化工具 —— 链接预测 & 关系预测 & 三元组分类 & 实体相似度

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

# ── 强制浅色风格 ──────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")

_STYLE_WHITE = "#ffffff"
_STYLE_LIGHT_GRAY = "#f5f5f5"
_STYLE_HEADER_BG = "#e8e8e8"
_STYLE_TEXT_DARK = "#222222"

matplotlib.rcParams.update(
    {
        "figure.facecolor": _STYLE_WHITE,
        "figure.edgecolor": _STYLE_WHITE,
        "axes.facecolor": _STYLE_LIGHT_GRAY,
        "axes.edgecolor": "#cccccc",
        "text.color": _STYLE_TEXT_DARK,
        "axes.labelcolor": _STYLE_TEXT_DARK,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "savefig.facecolor": _STYLE_WHITE,
        "savefig.transparent": False,
        "grid.alpha": 0.2,
    }
)

# ── CJK 字体 ──────────────────────────────────────────────────
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

# ── 调色板 ────────────────────────────────────────────────────
COLOR_TRAIN = "#2ecc71"
COLOR_VAL = "#3498db"
COLOR_TEST = "#e74c3c"
COLOR_POS = "#2ecc71"
COLOR_NEG = "#e74c3c"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

# ── FB15k-237 关系域名解析 ─────────────────────────────────────
_FB_MAJOR_DOMAINS = {
    "film", "tv", "music", "sports", "soccer", "american_football",
    "award", "location", "people", "organization", "education",
    "government", "military", "business", "media_common", "food",
    "book", "travel", "olympics", "broadcast", "computer",
    "medicine", "religion", "biology", "chemistry", "architecture",
    "transportation", "geography", "language", "fashion",
    "protected_sites",
}


def parse_fb_relation_domain(relation_name: str) -> str:
    """从 FB15k-237 关系名中提取主域。

    ``/film/film/language`` → ``film``
    ``/award/award_nominee/award_nominations./award/award_nomination/award`` → ``award``
    """
    parts = [p for p in relation_name.split("/") if p]
    if not parts:
        return "unknown"
    domain = parts[0]
    if domain in _FB_MAJOR_DOMAINS:
        return domain
    return "other"


def format_relation_name(name: str) -> str:
    """精简关系名：去除首尾斜杠，保留中间路径。

    ``/film/film/language`` → ``film/film/language``
    ``_hypernym`` → ``hypernym``
    """
    stripped = name.strip("/")
    if stripped.startswith("_"):
        stripped = stripped[1:]
    return stripped


def classify_relation_cardinality(tph: float, hpt: float) -> str:
    """根据 tph/hpt 统计量分类关系基数类型。

    Returns:
        "1-1", "1-N", "N-1", 或 "N-N"
    """
    if tph < 1.5 and hpt < 1.5:
        return "1-1"
    if tph >= 1.5 and hpt < 1.5:
        return "1-N"
    if tph < 1.5 and hpt >= 1.5:
        return "N-1"
    return "N-N"


def infer_entity_domain(
    entity_id: int,
    train_triples: Tensor,
    relation_domains: dict[int, str],
) -> str:
    """从实体参与的训练三元组推断其所属域。

    统计该实体作为头/尾出现的所有三元组所涉及的关系域，
    返回出现次数最多的域。
    """
    mask_h = train_triples[:, 0] == entity_id
    mask_t = train_triples[:, 2] == entity_id
    involved = torch.cat(
        [train_triples[mask_h, 1], train_triples[mask_t, 1]]
    )
    if len(involved) == 0:
        return "unknown"

    domain_counts: dict[str, int] = {}
    for r_id in involved.tolist():
        domain = relation_domains.get(int(r_id), "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    return max(domain_counts, key=domain_counts.get)


# ═══════════════════════════════════════════════════════════════
# 训练曲线
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(
    history: dict[str, list[float]],
    *,
    metric_key: str = "mrr",
    metric_label: str = "MRR",
    loss_mode: str = "train_loss_only",
) -> Figure:
    """训练曲线双子图：Loss（左）+ 性能指标（右）

    适配链接预测、关系预测、三元组分类三种训练器的输出。

    Args:
        history: 训练历史，含 ``loss`` 及 ``{metric_key}``, ``val_{metric_key}``
        metric_key: 性能指标键名（不含 val_ 前缀），如 ``"mrr"`` 或 ``"acc"``
        metric_label: 右图 y 轴标签
        loss_mode: ``"train_loss_only"`` — 仅绘制 train loss（链接预测无 val_loss）；
                   ``"val_loss"`` — 同时绘制 train + val loss（关系/三元组分类）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(_STYLE_WHITE)

    # eval_interval 对齐 —— history 中 val_* 键只在评估步记录
    eval_interval = _infer_eval_interval(history)
    loss_epochs = list(range(1, len(history.get("loss", [])) + 1))

    # ── Loss ──
    ax1.plot(loss_epochs, history["loss"], label="Train Loss", linewidth=1.2, color=COLOR_TRAIN)
    if loss_mode == "val_loss" and "val_loss" in history:
        val_epochs = [eval_interval * (i + 1) for i in range(len(history["val_loss"]))]
        ax1.plot(val_epochs, history["val_loss"], label="Val Loss", linewidth=1.2, color=COLOR_VAL)
        best_idx = int(np.argmin(history["val_loss"]))
        best_epoch = val_epochs[best_idx]
        ax1.axvline(
            x=best_epoch, color="gray", linestyle="--", alpha=0.5,
            label=f"Best (epoch {best_epoch})",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss 曲线")

    # ── Metric ──
    train_key = metric_key
    val_key = f"val_{metric_key}"
    if train_key in history:
        ax2.plot(
            [eval_interval * (i + 1) for i in range(len(history[train_key]))],
            history[train_key],
            label=f"Train {metric_label}",
            linewidth=1.2,
            color=COLOR_TRAIN,
        )
    if val_key in history:
        val_epochs = [eval_interval * (i + 1) for i in range(len(history[val_key]))]
        ax2.plot(
            val_epochs, history[val_key], label=f"Val {metric_label}",
            linewidth=1.2, color=COLOR_VAL,
        )
        best_idx = int(np.argmax(history[val_key]))
        best_val = history[val_key][best_idx]
        best_epoch = val_epochs[best_idx]
        ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax2.annotate(
            f"{best_val:.4f}",
            xy=(best_epoch, best_val),
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
                val_epochs[:min_len],
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


def _infer_eval_interval(history: dict[str, list[float]]) -> int:
    """推断 eval_interval：loss 长度与 val_* 长度之比。"""
    n_loss = len(history.get("loss", []))
    val_keys = [k for k in history if k.startswith("val_")]
    if not val_keys:
        return 1
    n_val = len(history[val_keys[0]])
    if n_val <= 1:
        return n_loss
    return max(1, n_loss // max(1, n_val))


# ═══════════════════════════════════════════════════════════════
# 三元组统计
# ═══════════════════════════════════════════════════════════════

def plot_triple_split_summary(dataset) -> Figure:
    """三元组划分可视化：各 split 三元组数量柱状图 + 摘要表

    Args:
        dataset: ``BaseKGDataset`` 实例（含 train_triples / val_triples / test_triples）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(_STYLE_WHITE)

    splits = ["Train", "Val", "Test"]
    counts = [
        len(dataset.train_triples),
        len(dataset.val_triples),
        len(dataset.test_triples),
    ]
    colors = [COLOR_TRAIN, COLOR_VAL, COLOR_TEST]

    # 左：柱状图
    x = np.arange(len(splits))
    bars = ax1.bar(x, counts, color=colors, alpha=0.85, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("三元组数量")
    ax1.set_title("各数据集三元组分布")
    ax1.grid(True, alpha=0.2, axis="y")

    # 右：摘要表
    ax2.axis("off")
    table_data = [
        ["实体数", f"{dataset.num_entities:,}"],
        ["关系数", f"{dataset.num_relations:,}"],
        ["训练集三元组", f"{counts[0]:,}"],
        ["验证集三元组", f"{counts[1]:,}"],
        ["测试集三元组", f"{counts[2]:,}"],
        ["总三元组", f"{sum(counts):,}"],
    ]
    table = ax2.table(
        cellText=table_data,
        colLabels=["指标", "数值"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(_STYLE_HEADER_BG)
            cell.set_text_props(weight="bold", color=_STYLE_TEXT_DARK)
        else:
            cell.set_facecolor(_STYLE_WHITE)
            cell.set_text_props(color=_STYLE_TEXT_DARK)
    ax2.set_title("数据集摘要", pad=15)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 关系级性能分析
# ═══════════════════════════════════════════════════════════════

def plot_relation_performance(
    relation_names: list[str],
    metrics: list[float],
    *,
    title: str = "各关系预测性能",
    metric_name: str = "MRR",
    top_n: int = 15,
) -> Figure:
    """各关系性能水平柱状图：top-N（绿）+ bottom-N（红）

    Args:
        relation_names: 关系显示名列表
        metrics: 各关系指标值
        title: 图表标题
        metric_name: 指标名称
        top_n: 显示前 N 和后 N 个
    """
    n = len(relation_names)
    if n <= top_n * 2:
        # 关系总数不够 top_n*2，全部显示
        indices = list(range(n))
        show_all = True
    else:
        sorted_idx = np.argsort(metrics)
        bottom = sorted_idx[:top_n].tolist()
        top = sorted_idx[-top_n:][::-1].tolist()
        indices = bottom + top
        show_all = False

    names = [relation_names[i] for i in indices]
    vals = [metrics[i] for i in indices]
    bar_colors = [
        COLOR_NEG if i < min(top_n, len(indices) - top_n) else COLOR_POS
        for i in range(len(indices))
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(indices) * 0.35)))
    fig.patch.set_facecolor(_STYLE_WHITE)

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric_name)
    ax.invert_yaxis()

    if not show_all:
        ax.axhline(y=top_n - 0.5, color="gray", linestyle="--", alpha=0.4)
        ax.text(
            0.98, top_n - 1, "┄  bottom", transform=ax.get_yaxis_transform(),
            fontsize=7, color=COLOR_NEG, ha="right", va="bottom",
        )
        ax.text(
            0.98, top_n, "┄  top", transform=ax.get_yaxis_transform(),
            fontsize=7, color=COLOR_POS, ha="right", va="top",
        )

    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def plot_relation_category_analysis(
    categories: dict[str, list[float]],
    *,
    metric_name: str = "MRR",
) -> Figure:
    """按关系类别（域）汇总的性能柱状图：均值 ± 标准差

    Args:
        categories: 域名 → 该域下所有关系的指标值列表
        metric_name: 指标名称
    """
    # 按均值排序
    cat_means = {
        cat: (np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
        for cat, vals in categories.items()
    }
    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1][0], reverse=True)
    cat_names = [c[0] for c in sorted_cats]
    means = [c[1][0] for c in sorted_cats]
    stds = [c[1][1] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(10, max(5, len(cat_names) * 0.4)))
    fig.patch.set_facecolor(_STYLE_WHITE)

    y_pos = np.arange(len(cat_names))
    bars = ax.barh(
        y_pos, means, xerr=stds,
        color=COLOR_VAL, alpha=0.75, edgecolor="white",
        capsize=3,
    )
    # 标注计数值
    for i, (cat, vals) in enumerate(
        [(c[0], categories[c[0]]) for c in sorted_cats]
    ):
        ax.text(
            means[i] + stds[i] + max(means) * 0.01,
            i,
            f"n={len(vals)}",
            fontsize=7,
            va="center",
            color="#555555",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_names, fontsize=9)
    ax.set_xlabel(metric_name)
    ax.invert_yaxis()
    ax.set_title(f"各领域 {metric_name}（均值 ± 标准差）")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 混淆矩阵
# ═══════════════════════════════════════════════════════════════

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
        title = "混淆矩阵（行归一化 → 召回率）"
    else:
        cm_display = cm
        fmt = "d"
        title = "混淆矩阵（绝对计数）"

    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 0.9)))
    fig.patch.set_facecolor(_STYLE_WHITE)
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


# ═══════════════════════════════════════════════════════════════
# t-SNE 嵌入可视化
# ═══════════════════════════════════════════════════════════════

def plot_tsne_embeddings(
    embeddings: Tensor | NDArray[np.floating],
    labels: Tensor | NDArray[np.integer],
    *,
    title: str = "t-SNE 实体嵌入可视化",
    color_map: str = "tab10",
    label_names: dict[int, str] | None = None,
    sample_size: int = 3000,
) -> Figure:
    """t-SNE 降维 + 按类别/度数着色的散点图

    针对 KGE 实体嵌入场景：无 train/val/test mask 概念，
    支持类别标签着色（如实体域）或连续值着色（如实体度数）。

    Args:
        embeddings: 实体嵌入 [N, D]
        labels: 标签 [N]，可以是离散类别（int）或连续度数（float）→ int
        title: 图标题
        color_map: matplotlib colormap 名称
        label_names: 类别 ID → 显示名映射（离散标签时使用）
        sample_size: 若 N 超过此值，随机采样
    """
    from sklearn.manifold import TSNE

    if isinstance(embeddings, Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = np.asarray(embeddings)
    if isinstance(labels, Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    # 采样
    N = len(embeddings_np)
    if N > sample_size:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, size=sample_size, replace=False)
        embeddings_np = embeddings_np[sample_idx]
        labels_np = labels_np[sample_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800)
    reduced = tsne.fit_transform(embeddings_np)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(_STYLE_WHITE)
    unique_labels = np.unique(labels_np)
    cmap = plt.colormaps[color_map]
    n_unique = len(unique_labels)

    # 判断是离散类别还是连续值：类别数少时用离散着色，多时用连续 colorbar
    use_discrete = n_unique <= 20

    if use_discrete and n_unique > 1:
        for i, label_val in enumerate(unique_labels):
            subset = labels_np == label_val
            if not subset.any():
                continue
            display_name = (
                label_names.get(int(label_val), f"Class {label_val}")
                if label_names
                else f"Class {int(label_val)}"
            )
            ax.scatter(
                reduced[subset, 0],
                reduced[subset, 1],
                c=[cmap(i / max(1, n_unique - 1))],
                marker="o",
                s=10,
                alpha=0.5,
                edgecolors="none",
                label=display_name,
            )
        ax.legend(fontsize=7, markerscale=1.5, loc="upper right")
    else:
        scatter = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels_np,
            cmap=cmap,
            marker="o",
            s=8,
            alpha=0.5,
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("度数", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# ROC / PR 曲线
# ═══════════════════════════════════════════════════════════════

def plot_roc_pr_curves(
    y_true: NDArray[np.floating],
    y_score: NDArray[np.floating],
) -> Figure:
    """ROC + PR 曲线双子图

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测 logits（未经 sigmoid）
    """
    from sklearn.metrics import (
        RocCurveDisplay,
        PrecisionRecallDisplay,
        roc_auc_score,
        average_precision_score,
    )

    y_true_b = y_true.astype(int)
    y_prob = 1 / (1 + np.exp(-y_score))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(_STYLE_WHITE)

    auc = roc_auc_score(y_true_b, y_prob)
    RocCurveDisplay.from_predictions(
        y_true_b, y_prob, ax=ax1, name=f"KGE (AUC={auc:.4f})"
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax1.set_title("ROC 曲线")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    ap = average_precision_score(y_true_b, y_prob)
    PrecisionRecallDisplay.from_predictions(
        y_true_b, y_prob, ax=ax2, name=f"KGE (AP={ap:.4f})"
    )
    baseline = y_true_b.mean()
    ax2.axhline(
        y=baseline, color="gray", linestyle="--", alpha=0.5,
        label=f"Baseline ({baseline:.3f})",
    )
    ax2.set_title("Precision-Recall 曲线")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 预测分数分布
# ═══════════════════════════════════════════════════════════════

def plot_prediction_histogram(
    y_true: NDArray[np.floating],
    y_score: NDArray[np.floating],
    *,
    bins: int = 50,
) -> Figure:
    """预测分数分布直方图（正/负样本分色）

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测 logits（未经 sigmoid）
        bins: 直方图柱数
    """
    y_true_b = y_true.astype(int)
    y_prob = 1 / (1 + np.exp(-y_score))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(_STYLE_WHITE)
    ax.hist(
        y_prob[y_true_b == 1],
        bins=bins,
        alpha=0.6,
        color=COLOR_POS,
        label=f"正样本 (n={int(y_true_b.sum())})",
        density=True,
    )
    ax.hist(
        y_prob[y_true_b == 0],
        bins=bins,
        alpha=0.6,
        color=COLOR_NEG,
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


# ═══════════════════════════════════════════════════════════════
# 最近邻表格
# ═══════════════════════════════════════════════════════════════

def plot_nearest_neighbors_table(
    query_names: list[str],
    neighbor_names_list: list[list[str]],
    similarities_list: list[list[float]],
    *,
    top_k: int = 10,
) -> Figure:
    """实体最近邻表格

    Args:
        query_names: 查询实体显示名列表
        neighbor_names_list: 对每个查询，top-K 邻居显示名列表
        similarities_list: 对每个查询，top-K 余弦相似度列表
        top_k: 邻居数量
    """
    rows: list[list[str]] = []
    for q_name, nbrs, sims in zip(query_names, neighbor_names_list, similarities_list):
        rows.append([q_name, "", ""])
        for rank, (nbr, sim) in enumerate(zip(nbrs, sims)):
            rows.append([f"  #{rank + 1}", nbr, f"{sim:.4f}"])

    fig, ax = plt.subplots(
        figsize=(12, max(3, 0.45 * len(rows) + 1.8)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(_STYLE_WHITE)
    ax.set_facecolor(_STYLE_WHITE)
    ax.set_axis_off()

    table = ax.table(
        cellText=rows,
        colLabels=["查询实体", "邻居实体", "余弦相似度"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(_STYLE_HEADER_BG)
            cell.set_text_props(weight="bold", color=_STYLE_TEXT_DARK)
        elif col == 0 and "  #" not in str(cell.get_text().get_text()):
            # 查询实体行高亮
            cell.set_facecolor("#e8f5e9")
            cell.set_text_props(weight="bold", color=_STYLE_TEXT_DARK)
        else:
            cell.set_facecolor(_STYLE_WHITE)
            cell.set_text_props(color=_STYLE_TEXT_DARK)

    ax.set_title(
        f"实体最近邻检索（Top-{top_k} 余弦相似度）",
        fontweight="bold",
        pad=20,
        color=_STYLE_TEXT_DARK,
    )
    return fig
