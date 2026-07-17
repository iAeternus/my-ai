import sys
import logging
from typing import Any, cast

from gnn.config import from_cli, parse_args
from gnn.utils.paths import resolve_config
from gnn.config.schema import TaskType
from gnn.datasets import load_planetoid
from gnn.datasets.splitter import split_link_prediction_data
from gnn.experiments import ExperimentManager
from gnn.models.builder import build_model
from gnn.trainer.factory import create_trainer
from core.utils import get_device, seed_everything, setup_logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def main() -> None:
    # 配置
    args = parse_args()
    config_path: str | None = (
        str(resolve_config(args.config)) if args.config else None
    )
    cfg = from_cli(config_path, overrides=vars(args))

    # 加载数据
    data = load_planetoid(cfg.dataset.name, cfg.dataset.root)

    # 输入特征归一化（Row-wise L2，GNN 标准预处理）
    data.x = F.normalize(cast(Tensor, data.x), p=2, dim=1)

    # 实验目录
    exp = ExperimentManager(cfg)
    multi_seed = len(cfg.experiment.seeds) > 1
    if multi_seed:
        exp.setup_multi()
    else:
        exp.setup()
    setup_logging(exp.log_dir)

    # 设备
    device = get_device(cfg.runtime.device)

    # 数据处理
    if cfg.task == TaskType.LINK_PREDICTION:
        train_data, val_data, test_data = split_link_prediction_data(data)
    else:
        train_data = val_data = test_data = data

    x: Tensor = cast(Tensor, data.x)
    y: Tensor = cast(Tensor, data.y)
    edge_index: Tensor = cast(Tensor, data.edge_index)
    num_classes = int(y.max().item()) + 1

    results: dict[int, dict[str, Any]] = {}
    for seed in cfg.experiment.seeds:
        seed_everything(seed)

        run_dir = exp.seed_run_dir(seed) if multi_seed else exp.root_dir

        # 构建模型
        model = build_model(cfg, num_features=x.shape[1], num_classes=num_classes)

        # 训练
        trainer = create_trainer(cfg, model, device)
        history = trainer.train(
            train_data, val_data, test_data, run_dir / "checkpoints"
        )

        # 保存产物 —— 多 seed 时各 seed 独立保存
        if multi_seed:
            exp.save_history(history, run_dir=run_dir)
            exp.plot_metrics(history, run_dir=run_dir)
        else:
            exp.save_history(history)

        # 提取最终测试指标
        results[seed] = {
            k: v[-1]
            for k, v in history.items()
            if k.startswith("test_") and v and isinstance(v, list)
        }

    # 汇总
    if multi_seed:
        key = "test_acc" if cfg.task == TaskType.NODE_CLASSIFICATION else "test_auc"
        exp.summarize_seeds({s: r.get(key, 0) for s, r in results.items()})
    else:
        seed = cfg.experiment.seeds[0]
        metrics = results.get(seed, {})
        if not metrics:
            # fallback: 从 history 中取最佳 val 指标
            monitor = cfg.train.early_stopping.monitor
            if monitor in history:
                best_value = (
                    max(history[monitor])
                    if "acc" in monitor or "auc" in monitor
                    else min(history[monitor])
                )
                metrics = {f"best_{monitor}": best_value}
        exp.save_metrics(metrics)
        exp.plot_metrics(history)
    logger.info("原始命令: %s", sys.argv)
    logger.info("实验完成: %s", exp.root_dir)


if __name__ == "__main__":
    main()
