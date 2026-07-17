from __future__ import annotations
import logging

from kge.utils.paths import PROJECT_ROOT
from kge.config import from_cli, parse_args
from kge.datasets import KGDataModule, load_dataset
from kge.experiments import ExperimentManager
from kge.models import build_model
from kge.trainer import create_trainer
from kge.utils import get_device, seed_everything, setup_logging
from kge.utils.typing import dict_pop_or_default

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    # CLI 参数
    cli_overrides = parse_args(argv)
    config_path = dict_pop_or_default(
        cli_overrides,
        "config",
        str(PROJECT_ROOT / "config" / "link_prediction-baseline.yaml"),
    )

    # 配置
    cfg = from_cli(config_path, cli_overrides)

    # 日志
    setup_logging()

    # 设备
    device = get_device(cfg.runtime.device)
    logger.info(f"Device: {device}")

    # 实验
    exp_mgr = ExperimentManager(cfg)

    for seed in cfg.experiment.seeds:
        logger.info(f"=== Seed {seed} ===")
        seed_everything(seed)

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

        # 实验目录
        if exp_mgr.is_multi:
            exp_mgr.setup_multi()
            run_dir = exp_mgr.seed_run_dir(seed)
            setup_logging(run_dir / "logs")
        else:
            exp_mgr.setup()
            run_dir = exp_mgr.root_dir
            setup_logging(exp_mgr.log_dir)

        # 训练
        history = trainer.train(run_dir / "checkpoints")

        # 保存
        exp_mgr.save_config()
        exp_mgr.save_history(history, run_dir=run_dir)
        exp_mgr.plot_metrics(history, run_dir=run_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()
