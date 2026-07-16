"""CLI 参数解析"""

from __future__ import annotations
import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KGE — Knowledge Graph Embedding 训练与评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["link_prediction", "relation_prediction", "triple_classification"],
        help="任务类型",
    )

    # 数据
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称")
    parser.add_argument("--root", type=str, default=None, help="数据集根目录")
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="训练 batch 大小",
    )
    parser.add_argument(
        "--negative-samples",
        dest="negative_samples",
        type=int,
        default=None,
        help="负采样数",
    )

    # 模型（--model 是 --encoder 的别名，兼容 GNN 风格）
    parser.add_argument("--encoder", type=str, default=None, help="KGE encoder 名称")
    parser.add_argument(
        "--model", type=str, default=None, dest="encoder", help="--encoder 的别名"
    )
    parser.add_argument("--head", type=str, default=None, help="预测头名称")
    parser.add_argument(
        "--embedding-dim", dest="embedding_dim", type=int, default=None, help="嵌入维度"
    )
    parser.add_argument("--gamma", type=float, default=None, help="Margin 值")
    parser.add_argument(
        "--p-norm", dest="p_norm", type=int, default=None, help="Lp 范数 (1 或 2)"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, default=None, help="Head 隐层维度"
    )
    parser.add_argument(
        "--hidden-dropout",
        dest="hidden_dropout",
        type=float,
        default=None,
        help="Dropout 概率",
    )

    # 训练
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument(
        "--weight-decay", dest="weight_decay", type=float, default=None, help="权重衰减"
    )
    parser.add_argument(
        "--loss-type",
        dest="loss_type",
        type=str,
        default=None,
        choices=["margin_ranking", "bce", "cross_entropy"],
        help="损失函数类型",
    )
    parser.add_argument(
        "--margin", type=float, default=None, help="Margin Ranking Loss 的 margin"
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        type=float,
        default=None,
        help="标签平滑",
    )
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        type=int,
        default=None,
        help="评估间隔 (epoch)",
    )
    parser.add_argument("--patience", type=int, default=None, help="早停 patience")

    # 实验
    parser.add_argument(
        "--name-prefix", dest="name_prefix", type=str, default=None, help="实验名称前缀"
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", type=str, default=None, help="产物输出根目录"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="随机种子列表"
    )

    # 运行时
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu/cuda/auto)")
    parser.add_argument(
        "--compile", type=str, default=None, help="torch.compile (true/false/auto)"
    )

    return parser


def parse_args(argv: list[str] | None = None) -> dict[str, object]:
    """解析 CLI 参数，返回非 None 值的扁平字典"""
    parser = create_parser()
    args = parser.parse_args(argv)
    return {k: v for k, v in vars(args).items() if v is not None}
