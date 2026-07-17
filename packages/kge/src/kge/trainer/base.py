from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from kge.config.schema import Config
from kge.datasets.data_module import KGDataModule, KGBatch
from kge.models.builder import KGEModel
from core.utils import (
    EarlyStopping,
    MONITOR_MODES,
    CheckpointManager,
    dict_pop_or_default,
)

logger = logging.getLogger(__name__)

_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class BaseTrainer(ABC):
    """KGE 训练器基类

    子类需实现: _train_step, _eval, _monitor_mode, _monitor_metric
    """

    def __init__(
        self,
        cfg: Config,
        model: KGEModel,
        data_module: KGDataModule,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model: KGEModel = model.to(device)
        self.data_module = data_module
        self.device = device

        # compile
        compile_mode = cfg.runtime.compile
        should_compile = (compile_mode == "auto" and device.type == "cuda") or str(
            compile_mode
        ).lower() == "true"
        if should_compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        # optimizer
        opt_cfg = cfg.optimizer
        opt_params: dict = dict(opt_cfg.params)
        lr = dict_pop_or_default(opt_params, "lr", cfg.train.lr)
        weight_decay = dict_pop_or_default(
            opt_params,
            "weight_decay",
            cfg.train.weight_decay,
        )
        optimizer_cls = _OPTIMIZERS.get(opt_cfg.name, torch.optim.Adam)
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **opt_params,
        )

        self.scheduler = None
        self._best_score: float | None = None
        self._checkpoint_mgr: CheckpointManager | None = None

    @abstractmethod
    def _train_step(self, batch: KGBatch) -> tuple[float, dict[str, float]]:
        """单步训练，返回 (loss_scalar, metrics_dict)"""
        ...

    @abstractmethod
    @torch.no_grad()
    def _eval(self, split: str) -> tuple[float, dict[str, float]]:
        """评估指定 split ('val' / 'test')，返回 (loss, metrics_dict)"""
        ...

    @property
    @abstractmethod
    def _monitor_mode(self) -> str:
        """ "max" 或 "min" """
        ...

    @property
    @abstractmethod
    def _monitor_metric(self) -> str:
        """如 "val_mrr" """
        ...

    def train(self, checkpoint_dir: Path) -> dict[str, list[float]]:
        """训练主循环"""
        history: dict[str, list[float]] = {"loss": [], "epoch_time": []}
        e = self.cfg.train.early_stopping
        early_stop = EarlyStopping(
            patience=e.patience,
            mode=MONITOR_MODES[e.monitor],
            min_delta=e.min_delta,
        )

        self._checkpoint_mgr = CheckpointManager(checkpoint_dir)

        train_loader = self.data_module.train_dataloader()

        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            t0 = time.perf_counter()
            epoch_loss = 0.0
            epoch_metrics: dict[str, float] = {}

            for batch in train_loader:
                loss, step_metrics = self._train_step(batch)
                epoch_loss += loss
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            n_batches = len(train_loader)
            epoch_loss /= n_batches
            for k in epoch_metrics:
                epoch_metrics[k] /= n_batches

            elapsed = time.perf_counter() - t0
            history["loss"].append(epoch_loss)
            history["epoch_time"].append(elapsed)

            logger.info(
                f"Epoch {epoch + 1:3d}/{self.cfg.train.epochs} | "
                f"loss={epoch_loss:.4f} | time={elapsed:.1f}s"
            )

            # 评估
            if (epoch + 1) % self.cfg.train.eval_interval == 0:
                _, val_metrics = self._eval("val")
                for k, v in val_metrics.items():
                    key = f"val_{k}"
                    history.setdefault(key, []).append(v)

                monitor_key = self._monitor_metric
                monitor_val = val_metrics.get(monitor_key.replace("val_", ""), 0.0)
                logger.info(
                    f"  Eval | {monitor_key}={monitor_val:.4f} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )

                improved = early_stop.improved
                should_stop = early_stop.step(monitor_val)
                if self._checkpoint_mgr is not None:
                    self._checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        epoch,
                        monitor_val,
                        mode=self._monitor_mode,
                        min_delta=e.min_delta,
                        history=history,
                    )

                if should_stop:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型并测试
        if self._checkpoint_mgr is not None:
            self._checkpoint_mgr.load(self.model)
        test_loss, test_metrics = self._eval("test")
        for k, v in test_metrics.items():
            history[f"test_{k}"] = [v]
        logger.info(
            f"Test | " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        )

        return history
