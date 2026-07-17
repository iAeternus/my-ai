from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch import nn


class CheckpointManager:
    """管理latest.pt和best.pt"""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dir / "latest.pt"
        self.best_path = self.dir / "best.pt"
        self._best_metric: float | None = None

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        history: dict[str, list[float]] | None = None,
        *,
        mode: str = "max",
        min_delta: float = 1e-6,
    ) -> bool:
        """保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch
            metric: 监控指标值
            scheduler: 学习率调度器（可选）
            history: 训练历史（可选）
            mode: "max" 或 "min"

        Returns:
            True 如果是最佳并保存了 best.pt
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if history is not None:
            checkpoint["history"] = history

        torch.save(checkpoint, self.latest_path)

        is_best = False
        if self._best_metric is None:
            is_best = True
        elif mode == "max" and metric > self._best_metric + min_delta:
            is_best = True
        elif mode == "min" and metric < self._best_metric - min_delta:
            is_best = True

        if is_best:
            self._best_metric = metric
            torch.save(checkpoint, self.best_path)

        return is_best

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        *,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """加载检查点

        Args:
            model: 模型（原地加载权重）
            optimizer: 优化器（原地加载状态）
            scheduler: 调度器（原地加载状态）
            path: 指定路径，默认加载 best.pt

        Returns:
            包含 epoch, metric, history 的字典
        """
        load_path = Path(path) if path else self.best_path
        if not load_path.exists():
            raise FileNotFoundError(f"检查点不存在: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metric": checkpoint.get("metric", 0.0),
            "history": checkpoint.get("history", {}),
        }
