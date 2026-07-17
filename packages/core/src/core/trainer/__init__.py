"""提供通用训练循环抽象和生命周期回调协议"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Generic, TypeVar

import torch
import torch.nn as nn

from core.utils.checkpoint import CheckpointManager
from core.utils.early_stopping import EarlyStopping, MONITOR_MODES
from core.utils.device import get_device
from core.utils.registry import Registry

logger = logging.getLogger(__name__)

TBatch = TypeVar("TBatch")


class Callback(ABC):
    """训练生命周期钩子

    所有钩子均有默认空实现，子类按需覆盖
    钩子调用顺序：

    ::

        on_train_begin
          for epoch:
            on_epoch_begin
              for batch:
                on_batch_begin -> _train_step -> on_batch_end
            on_epoch_end
            on_eval_begin -> _eval -> on_eval_end
        on_train_end
    """

    def on_train_begin(self, trainer: BaseTrainer) -> None:
        """训练开始前调用一次"""

    def on_train_end(
        self, trainer: BaseTrainer, history: dict[str, list[float]]
    ) -> None:
        """训练结束后调用一次"""

    def on_epoch_begin(self, trainer: BaseTrainer, epoch: int) -> None:
        """每个 epoch 开始时调用"""

    def on_epoch_end(
        self, trainer: BaseTrainer, epoch: int, logs: dict[str, Any]
    ) -> None:
        """每个 epoch 结束时调用"""

    def on_batch_begin(self, trainer: BaseTrainer, batch_idx: int) -> None:
        """每个 batch 开始时调用"""

    def on_batch_end(self, trainer: BaseTrainer, batch_idx: int, loss: float) -> None:
        """每个 batch 结束时调用"""

    def on_eval_begin(self, trainer: BaseTrainer, split: str) -> None:
        """评估阶段开始时调用，``split`` 为 ``"val"`` 或 ``"test"``"""

    def on_eval_end(
        self, trainer: BaseTrainer, split: str, metrics: dict[str, float]
    ) -> None:
        """评估阶段结束时调用"""


class BaseTrainer(ABC, Generic[TBatch]):
    """通用 ML 训练器基类（Template Method 模式）

    子类只需实现 4 个抽象方法：``_train_step``, ``_eval``,
    ``_monitor_mode``, ``_monitor_metric``

    回调系统处理：早停、checkpoint、AMP、梯度累积、lr scheduler 等横切关注点
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: "torch.optim.Optimizer",
        *,
        device: str | torch.device = "auto",
        compile: str | bool = "auto",
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.device = get_device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self._callbacks = callbacks or []

        # torch.compile
        should_compile = False
        if compile == "auto":
            should_compile = torch.cuda.is_available()
        elif compile is True or str(compile).lower() == "true":
            should_compile = True

        if should_compile:
            self.model.compile()

    @abstractmethod
    def _train_step(self, batch: TBatch) -> tuple[float, dict[str, float]]:
        """执行一步训练

        Returns:
            (loss_value, metrics_dict) — 如 ``(0.5, {"acc": 0.8})``
        """
        ...

    @abstractmethod
    def _eval(self, split: str) -> tuple[float, dict[str, float]]:
        """在 ``split``（``"val"`` / ``"test"``）上评估

        Returns:
            (loss_value, metrics_dict)
        """
        ...

    @property
    @abstractmethod
    def _monitor_mode(self) -> str:
        """监控模式：``"max"``（越大越好）或 ``"min"``（越小越好）"""
        ...

    @property
    @abstractmethod
    def _monitor_metric(self) -> str:
        """监控指标名（如 ``"val_acc"``, ``"val_mrr"``）"""
        ...

    def train(
        self,
        train_loader: Iterable[TBatch],
        *,
        epochs: int = 100,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        eval_interval: int = 1,
        eval_fn: Callable[[str], tuple[float, dict[str, float]]] | None = None,
    ) -> dict[str, list[float]]:
        """运行完整训练循环

        Args:
            train_loader: 训练数据迭代器
            epochs: 最大训练轮数
            early_stopping: 早停实例（可选）
            checkpoint_manager: Checkpoint 管理器（可选）
            eval_interval: 每隔多少 epoch 评估一次
            eval_fn: 可选的外部评估函数，签名 ``(split) -> (loss, metrics)``
                     为 ``None`` 时使用 ``self._eval``

        Returns:
            history 字典（key -> epoch 值列表）
        """
        # 如果没有传入外部 eval_fn，使用自身的 _eval
        _do_eval = eval_fn or self._eval

        # 回调：on_train_begin
        for cb in self._callbacks:
            cb.on_train_begin(self)

        history: dict[str, list[float]] = {}
        early_stop_triggered = False

        for epoch in range(1, epochs + 1):
            # 回调：on_epoch_begin
            for cb in self._callbacks:
                cb.on_epoch_begin(self, epoch)

            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics: dict[str, float] = {}

            for batch_idx, batch in enumerate(train_loader):
                for cb in self._callbacks:
                    cb.on_batch_begin(self, batch_idx)

                loss, step_metrics = self._train_step(batch)

                for cb in self._callbacks:
                    cb.on_batch_end(self, batch_idx, loss)

                epoch_loss += loss
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            n_batches = batch_idx + 1
            epoch_loss /= n_batches
            epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

            # 记录训练指标
            self._append_log(history, "loss", epoch_loss)
            for k, v in epoch_metrics.items():
                self._append_log(history, k, v)

            # 评估阶段
            if epoch % eval_interval == 0:
                for split in ("val", "test"):
                    try:
                        for cb in self._callbacks:
                            cb.on_eval_begin(self, split)
                        val_loss, val_metrics = _do_eval(split)
                        for cb in self._callbacks:
                            cb.on_eval_end(
                                self, split, {**val_metrics, "loss": val_loss}
                            )

                        self._append_log(history, f"{split}_loss", val_loss)
                        for k, v in val_metrics.items():
                            self._append_log(history, f"{split}_{k}", v)
                    except Exception:
                        logger.debug("跳过 %s 评估", split, exc_info=True)

            # 日志
            monitor_val = history.get(self._monitor_metric, [None])[-1]
            log_parts = [f"Epoch {epoch:3d}/{epochs}"]
            log_parts.append(f"loss={epoch_loss:.4f}")
            if monitor_val is not None:
                log_parts.append(f"{self._monitor_metric}={monitor_val:.4f}")
            logger.info(" | ".join(log_parts))

            # 早停 + Checkpoint
            if early_stopping is not None and monitor_val is not None:
                should_stop = early_stopping.step(monitor_val)
                if checkpoint_manager is not None and early_stopping.improved:
                    self._save_checkpoint(checkpoint_manager, epoch, monitor_val)

                if should_stop:
                    logger.info("早停触发于 epoch %d", epoch)
                    early_stop_triggered = True

            # 回调：on_epoch_end
            epoch_logs = {
                "epoch": epoch,
                "loss": epoch_loss,
                **epoch_metrics,
            }
            for cb in self._callbacks:
                cb.on_epoch_end(self, epoch, epoch_logs)

            if early_stop_triggered:
                break

        # 恢复最佳模型
        if checkpoint_manager is not None:
            info = checkpoint_manager.load(self.model)
            if info:
                logger.info(
                    "加载最佳模型: epoch=%s, %s=%.4f",
                    info.get("epoch"),
                    self._monitor_metric,
                    info.get("metric", float("nan")),
                )

        # 回调：on_train_end
        for cb in self._callbacks:
            cb.on_train_end(self, history)

        return history

    @staticmethod
    def _append_log(history: dict[str, list[float]], key: str, value: float) -> None:
        history.setdefault(key, []).append(value)

    def _save_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        epoch: int,
        metric: float,
    ) -> None:
        """保存 checkpoint（子类可覆盖以添加 scheduler、history）。"""
        checkpoint_manager.save(
            self.model,
            self.optimizer,
            epoch,
            metric,
        )


class EarlyStoppingCallback(Callback):
    """将 ``EarlyStopping`` 包装为 Callback

    Usage::

        es = EarlyStopping(patience=30, mode="max")
        trainer = MyTrainer(model, opt, callbacks=[EarlyStoppingCallback(es)])
    """

    def __init__(self, early_stopping: EarlyStopping) -> None:
        super().__init__()
        self._es = early_stopping

    def on_epoch_end(
        self,
        trainer: BaseTrainer,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        metric = logs.get(trainer._monitor_metric.replace("val_", ""))
        if metric is not None:
            self._es.step(metric)


class CheckpointCallback(Callback):
    """将 ``CheckpointManager`` 包装为 Callback

    Usage::

        ckpt = CheckpointManager("outputs/checkpoints")
        trainer = MyTrainer(model, opt, callbacks=[CheckpointCallback(ckpt)])
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        *,
        save_best: bool = True,
        save_latest: bool = False,
    ) -> None:
        super().__init__()
        self._ckpt = checkpoint_manager
        self._save_latest = save_latest

    def on_epoch_end(
        self,
        trainer: BaseTrainer,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        metric = logs.get(trainer._monitor_metric.replace("val_", ""))
        if metric is not None:
            self._ckpt.save(
                trainer.model,
                trainer.optimizer,
                epoch,
                metric,
            )


OPTIMIZER_REGISTRY = Registry[type]("optimizer")

# 注册标准 PyTorch 优化器


OPTIMIZER_REGISTRY.register("adam")(torch.optim.Adam)
OPTIMIZER_REGISTRY.register("adamw")(torch.optim.AdamW)
OPTIMIZER_REGISTRY.register("sgd")(torch.optim.SGD)
