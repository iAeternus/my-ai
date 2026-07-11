import logging
from pathlib import Path
import time
from typing import cast

from iris_logistic_regression.config import IrisConfig
from iris_logistic_regression.utils.early_stopping import EarlyStopping
import torch
from torch.utils.data import DataLoader
from torch import nn

logger = logging.getLogger(__name__)


class IrisTrainer:
    def __init__(
        self,
        config: IrisConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
    ):
        self._config = config

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

        self._device = device

        self._model: nn.Module = model.to(self._device)

        if self._config.compile_model:
            try:
                compiled_model = torch.compile(self._model)
                self._model = cast(nn.Module, compiled_model)
                logger.info("模型已通过 torch.compile() 编译")
            except Exception as ex:
                logger.warning("torch.compile() 失败，使用 eager 模式: %s", ex)

        self._optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )

        self._criterion: nn.Module = nn.CrossEntropyLoss()

        self._best_val_loss = float("inf")
        self._save_path = self._config.output_dir / self._config.save_name
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict[str, list[float]]:
        history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
            "gpu_memory": [],
        }

        early_stopping = EarlyStopping(patience=self._config.patience, mode="min")

        for epoch in range(self._config.epochs):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self._optimizer.param_groups[0]["lr"]

            epoch_time = time.perf_counter() - epoch_start

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(self._save_path)

            history["loss"].append(train_loss)
            history["acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)
            if self._device.type == "cuda":
                history["gpu_memory"].append(
                    torch.cuda.max_memory_allocated(self._device) / 1024**3
                )
                torch.cuda.reset_peak_memory_stats(self._device)

            logger.info(
                "Epoch %3d/%d | "
                "训练 Loss: %.4f | 训练 Acc: %.4f | "
                "验证 Loss: %.4f | 验证 Acc: %.4f | "
                "LR: %.1e | 耗时: %.2fs",
                epoch,
                self._config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                epoch_time,
            )

            if early_stopping.step(val_loss):
                logger.info("早停触发于第 %d 轮", epoch)
                break

        self.load_checkpoint(self._save_path)

        test_acc = self.test()
        logger.info("测试准确率: %.4f", test_acc)

        return {**history, "test_acc": [test_acc]}

    def _train_epoch(self) -> tuple[float, float]:
        self._model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in self._train_loader:
            features = features.to(self._device)
            labels = labels.to(self._device)

            self._optimizer.zero_grad(set_to_none=True)

            output = self._model(features)
            loss = self._criterion(output, labels)
            loss.backward()
            self._optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (output.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

        return (
            total_loss / total_samples,
            total_correct / total_samples,
        )

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        return self._evaluate(self._val_loader)

    @torch.no_grad()
    def test(self) -> float:
        _, acc = self._evaluate(self._test_loader)
        return acc

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self._model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in dataloader:
            features = features.to(self._device)
            labels = labels.to(self._device)

            outputs = self._model(features)
            loss = self._criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

        return (
            total_loss / total_samples,
            total_correct / total_samples,
        )

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(self._model.state_dict(), path)
        logger.info("模型已保存至 %s", path)

    def load_checkpoint(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        state_dict = torch.load(
            model_path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(state_dict)
        logger.info("模型已加载: %s", model_path)

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """预测整个数据集"""
        preds: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []

        for features, _ in dataloader:
            logits = self._predict_logits(features)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            preds.append(pred.cpu())
            probs.append(prob.cpu())

        return (
            torch.cat(preds),
            torch.cat(probs),
        )

    @torch.no_grad()
    def predict_one(self, features: torch.Tensor) -> tuple[int, torch.Tensor]:
        """预测单个样本

        Args:
            features: 输入特征，形状为 ``(4,)`` 或 ``(1, 4)``

        Returns:
            一个二元组：
            - ``pred``：预测类别编号。
            - ``prob``：各类别预测概率，形状为 ``(3,)``
        """
        logits = self._predict_logits(features)
        prob = torch.softmax(logits, dim=1)
        pred = int(prob.argmax(dim=1).item())

        return pred, prob.squeeze(0).cpu()

    @torch.no_grad()
    def _predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        self._model.eval()

        features = features.to(self._device)

        if features.ndim == 1:
            features = features.unsqueeze(0)

        return self._model(features)
