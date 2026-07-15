import logging

from kge.config.loader import from_cli
from kge.config.parser import parse_args
from kge.datasets import load_dataset
from kge.utils.logging import setup_logging
from kge.utils.device import get_device

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    # CLI参数
    cli_overrides = parse_args(argv)
    config_path = cli_overrides.pop(
        "config", "config/link_prediction/transe-fb15k237.yaml"
    )

    # 配置
    cfg = from_cli(config_path, cli_overrides)

    # 日志
    setup_logging()

    # 设备
    device = get_device(cfg.runtime.device)
    logger.info(f"设备: {device}")

    # tmp
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.root)
    logger.info(f"数据集: {cfg.dataset.name} | entities={dataset.num_entities} relations={dataset.num_relations}")


if __name__ == "__main__":
    main()
