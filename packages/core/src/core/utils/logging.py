"""日志设置。"""

from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    *,
    level: int = logging.INFO,
) -> None:
    """配置同时输出到控制台和文件的日志

    Args:
        log_dir: 日志文件目录 (None = 仅控制台)
        level: 日志级别
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 文件
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "train.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
