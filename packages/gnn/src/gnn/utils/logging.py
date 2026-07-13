from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(save_dir: str | Path | None = None) -> None:
    """配置日志

    配置 Root Logger，日志输出到控制台，并可选保存到文件

    Args:
        save_dir: 若提供，则同时保存日志到 `save_dir / "train.log"`
    """
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [console_handler]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            save_dir / "train.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )
