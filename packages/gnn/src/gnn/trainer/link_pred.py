from typing import cast

from gnn.utils.metrics import binary_ap, binary_auc
import torch
from torch import Tensor, nn
from gnn.trainer.base import BaseTrainer
from gnn.config import Config
from gnn.utils.early_stopping import MONITOR_MODES


class LinkPredictionTrainer(BaseTrainer):
    """链接预测训练器"""

    def __init__(self, cfg: Config, model: nn.Module, device: torch.device) -> None:
        super().__init__(cfg, model, device)
        self.criterion = nn.BCEWithLogitsLoss()

    @property
    def _monitor_mode(self) -> str:
        return MONITOR_MODES.get(self.cfg.train.early_stopping.monitor, "max")

    @property
    def _monitor_metric(self) -> str:
        return self.cfg.train.early_stopping.monitor

    def _train_step(self, data) -> tuple[float, dict[str, float]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = cast(Tensor, data.x).to(self.device)
        edge_index = cast(Tensor, data.edge_index).to(self.device)
        edge_label_index = data.edge_label_index.to(self.device)
        edge_label = data.edge_label.to(self.device)

        logits = self.model(x, edge_index, edge_label_index=edge_label_index)
        loss = self.criterion(logits, edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        auc = binary_auc(edge_label.cpu(), logits.detach().cpu())
        return loss.item(), {"auc": auc}

    @torch.no_grad()
    def _eval(self, data, prefix: str = "val") -> tuple[float, dict[str, float]]:
        self.model.eval()

        x = cast(Tensor, data.x).to(self.device)
        edge_index = cast(Tensor, data.edge_index).to(self.device)
        edge_label_index = data.edge_label_index.to(self.device)
        edge_label = data.edge_label.to(self.device)

        logits = self.model(x, edge_index, edge_label_index=edge_label_index)
        loss = self.criterion(logits, edge_label).item()
        y_true = edge_label.cpu()
        y_score = logits.cpu()

        return loss, {
            f"{prefix}_loss": loss,
            f"{prefix}_auc": binary_auc(y_true, y_score),
            f"{prefix}_ap": binary_ap(y_true, y_score),
        }
