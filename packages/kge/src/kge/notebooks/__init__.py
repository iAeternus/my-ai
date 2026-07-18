"""KGE Notebook 可视化工具。

所有绘图函数均返回 ``matplotlib.figure.Figure``。
"""

from kge.notebooks.visualization import (
    # 工具函数
    classify_relation_cardinality,
    format_relation_name,
    infer_entity_domain,
    parse_fb_relation_domain,
    # 绘图函数
    plot_confusion_matrix,
    plot_nearest_neighbors_table,
    plot_prediction_histogram,
    plot_relation_category_analysis,
    plot_relation_performance,
    plot_roc_pr_curves,
    plot_training_curves,
    plot_triple_split_summary,
    plot_tsne_embeddings,
)

__all__ = [
    # 工具
    "parse_fb_relation_domain",
    "format_relation_name",
    "classify_relation_cardinality",
    "infer_entity_domain",
    # 绘图
    "plot_training_curves",
    "plot_triple_split_summary",
    "plot_relation_performance",
    "plot_relation_category_analysis",
    "plot_confusion_matrix",
    "plot_tsne_embeddings",
    "plot_roc_pr_curves",
    "plot_prediction_histogram",
    "plot_nearest_neighbors_table",
]
