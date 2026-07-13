from typing import cast

from gnn.config import from_cli, parse_args, validate_config
from gnn.config.schema import TaskType
from gnn.datasets import load_planetoid
from gnn.datasets.splitter import split_link_prediction_data
from gnn.experiments import ExperimentManager
from gnn.models.builder import build_model
from gnn.utils.device import get_device
from gnn.utils.logging import setup_logging

import torch
from torch import Tensor
from torch_geometric.data import Data


def main():
    # 配置
    args = parse_args()
    cfg = from_cli(args.config, overrides=vars(args))
    validate_config(cfg)

    # 加载数据
    data = load_planetoid(cfg.dataset.name, cfg.dataset.root)

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

    results: dict[int, dict] = {}
    for seed in cfg.experiment.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        run_dir = exp.seed_run_dir(seed) if multi_seed else exp.root_dir

        # 构建模型
        model = build_model(cfg, num_features=x.shape[1], num_classes=num_classes)

        # 训练
        ...


if __name__ == "__main__":
    main()
