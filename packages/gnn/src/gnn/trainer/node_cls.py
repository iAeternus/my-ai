from typing import cast, override


from gnn.utils.metrics import accuracy
import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from gnn.trainer.base import BaseTrainer
from gnn.config import Config
from core.utils import MONITOR_MODES


class NodeClassificationTrainer(BaseTrainer):
    """全批量节点分类训练器"""

    def __init__(self, cfg: Config, model: nn.Module, device: torch.device) -> None:
        super().__init__(cfg, model, device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    @property
    @override
    def _monitor_mode(self) -> str:
        return MONITOR_MODES[self.cfg.train.early_stopping.monitor]

    @property
    @override
    def _monitor_metric(self) -> str:
        return self.cfg.train.early_stopping.monitor

    @override
    def _train_step(self, data: Data) -> tuple[float, dict[str, float]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = cast(Tensor, data.x).to(self.device)
        edge_index = cast(Tensor, data.edge_index).to(self.device)
        y = cast(Tensor, data.y).to(self.device)

        logits = self.model(x, edge_index)
        loss = self.criterion(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        acc = accuracy(logits[data.train_mask], y[data.train_mask])
        return loss.item(), {"acc": acc}

    @override
    @torch.no_grad()
    def _eval(self, data: Data, prefix: str = "val") -> tuple[float, dict[str, float]]:
        self.model.eval()

        x = cast(Tensor, data.x).to(self.device)
        edge_index = cast(Tensor, data.edge_index).to(self.device)
        y = cast(Tensor, data.y).to(self.device)

        logits = self.model(x, edge_index)
        mask = data.val_mask if prefix == "val" else data.test_mask
        loss = self.criterion(logits[mask], y[mask]).item()

        acc = accuracy(logits[mask], y[mask])
        return loss, {f"{prefix}_loss": loss, f"{prefix}_acc": acc}
