from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import time
import torch
from torch import nn
from torch_geometric.data import Data

from gnn.config import Config
from gnn.utils.early_stopping import EarlyStopping
from gnn.utils.typing import pop_config

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """GNN 训练器基类

    子类需实现: _train_step, _eval, _monitor_mode, _monitor_metric
    """

    def __init__(self, cfg: Config, model: nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # compile
        compile_mode = cfg.runtime.compile
        should_compile = (compile_mode == "auto" and device.type == "cuda") or str(
            compile_mode
        ).lower() == "true"
        if should_compile:
            self.model = torch.compile(self.model)

        # optimizer
        opt_params: dict[str, Any] = dict(cfg.optimizer.params)
        lr = pop_config(opt_params, "lr", 0.01)
        weight_decay = pop_config(opt_params, "weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **opt_params,
        )

        # scheduler
        self.scheduler = None
        if cfg.scheduler.enabled and cfg.scheduler.name:
            sched_params: dict[str, Any] = dict(cfg.scheduler.params)
            if cfg.scheduler.name == "reduce_on_plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    **sched_params,
                )

        self._best_score: float | None = None
        self._checkpoint_path: Path | None = None

    @abstractmethod
    def _train_step(self, data: Data) -> tuple[float, dict[str, float]]:
        """返回 (loss, metrics_dict)"""
        ...

    @abstractmethod
    @torch.no_grad()
    def _eval(self, data: Data, prefix: str = "val") -> tuple[float, dict[str, float]]:
        """返回 (loss, metrics_dict)，prefix 为指标前缀 "val" 或 "test" """
        ...

    @property
    @abstractmethod
    def _monitor_mode(self) -> str:
        """ "min" 或 "max" """
        ...

    @property
    @abstractmethod
    def _monitor_metric(self) -> str:
        """如 "val_acc" 或 "val_auc" """
        ...

    def train(
        self,
        train_data: Data,
        val_data: Data,
        test_data: Data,
        checkpoint_dir: Path,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"loss": [], "epoch_time": []}
        early_stop = EarlyStopping(
            patience=self.cfg.train.early_stopping.patience,
            mode=self._monitor_mode,
        )
        self._checkpoint_path = checkpoint_dir / "best.pt"
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.train.epochs):
            # 训练
            t0 = time.perf_counter()

            train_loss, train_metrics = self._train_step(train_data)
            val_loss, val_metrics = self._eval(val_data, prefix="val")

            if self.scheduler is not None:
                monitor_val = val_metrics.get(self._monitor_metric, val_loss)
                self.scheduler.step(monitor_val)

            epoch_time = time.perf_counter() - t0

            # 记录
            history["loss"].append(train_loss)
            history["epoch_time"].append(epoch_time)
            for k, v in {**train_metrics, **val_metrics}.items():
                history.setdefault(k, []).append(v)

            log_parts = [
                f"Epoch {epoch:3d}/{self.cfg.train.epochs}",
                f"loss={train_loss:.4f}",
            ]
            for k, v in val_metrics.items():
                log_parts.append(f"{k}={v:.4f}")
            log_parts.append(f"time={epoch_time:.2f}s")
            logger.info(" | ".join(log_parts))

            # 保存与早停
            monitor_val = val_metrics.get(self._monitor_metric, val_loss)
            should_stop = early_stop.step(monitor_val)
            if early_stop.improved:
                torch.save(self.model.state_dict(), self._checkpoint_path)
            if should_stop:
                logger.info("早停触发于 epoch %d", epoch)
                break

        # 加载最佳模型并测试
        self._load_best()
        test_loss, test_metrics = self._eval(test_data, prefix="test")
        logger.info("测试结果: %s", test_metrics)
        return history

    def _load_best(self) -> None:
        if self._checkpoint_path and self._checkpoint_path.exists():
            state = torch.load(
                self._checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state)
