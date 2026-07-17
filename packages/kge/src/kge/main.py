"""KGE 实验入口。"""

from __future__ import annotations

import logging

from core.experiment import ExperimentManager, PlotSpec
from core.utils import (
    dict_pop_or_default,
    get_device,
    seed_everything,
    setup_logging,
)
from kge.config import from_cli, parse_args
from kge.datasets import KGDataModule, load_dataset
from kge.models import build_model
from kge.trainer import create_trainer
from kge.utils.paths import PACKAGE_ROOT

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    # CLI 参数
    cli_overrides = parse_args(argv)
    config_path: str | None = dict_pop_or_default(
        cli_overrides,
        "config",
        str(PACKAGE_ROOT / "config" / "link_prediction-baseline.yaml"),
    )

    # 配置
    cfg = from_cli(config_path, cli_overrides)

    # 日志
    setup_logging()

    # 设备
    device = get_device(cfg.runtime.device)
    logger.info(f"Device: {device}")

    # 实验目录 — [shared] 使用 core 统一 ExperimentManager
    exp_mgr = ExperimentManager(
        save_dir=cfg.experiment.save_dir,
        name_prefix=cfg.experiment.name_prefix,
        dir_segments=[cfg.dataset.name, cfg.model.encoder_name],
        seeds=cfg.experiment.seeds,
    )

    if exp_mgr.is_multi:
        exp_mgr.setup_multi()
    else:
        exp_mgr.setup()
    exp_mgr.save_config(cfg.to_dict())
    setup_logging(exp_mgr.log_dir)

    for seed in cfg.experiment.seeds:
        logger.info(f"=== Seed {seed} ===")
        seed_everything(seed)

        run_dir = exp_mgr.seed_run_dir(seed) if exp_mgr.is_multi else exp_mgr.root_dir

        # 数据
        dataset = load_dataset(cfg.dataset.name, cfg.dataset.root)
        data_module = KGDataModule(cfg, dataset)
        logger.info(
            f"Dataset: {cfg.dataset.name} | entities={data_module.num_entities} "
            f"relations={data_module.num_relations}"
        )

        # 模型
        model = build_model(cfg, data_module.num_entities, data_module.num_relations)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model: encoder={cfg.model.encoder_name} head={cfg.model.head_name} | "
            f"params={n_params:,}"
        )

        # Trainer
        trainer = create_trainer(cfg, model, data_module, device)

        # 训练
        history = trainer.train(run_dir / "checkpoints")

        # 保存
        exp_mgr.save_history(history, run_dir=run_dir)

        # 绘图 — [shared] PlotSpec 声明式替代硬编码分支
        mrr_keys = [k for k in history if "mrr" in k.lower()]
        hits_keys = [k for k in history if "hits" in k.lower()]
        specs = [
            PlotSpec(
                title="Loss", train_key="loss", val_key="val_loss", ylabel="Loss"
            ),
        ]
        if mrr_keys:
            specs.append(
                PlotSpec(title="MRR", multi_keys=mrr_keys, ylabel="MRR")
            )
        if hits_keys:
            specs.append(
                PlotSpec(title="Hits@K", multi_keys=hits_keys, ylabel="Hits@K")
            )
        exp_mgr.plot_metrics(history, specs=specs, run_dir=run_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()
