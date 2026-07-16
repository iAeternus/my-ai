from __future__ import annotations
import logging

import torch
import torch.nn.functional as F

from kge.config.schema import Config
from kge.datasets.data_module import KGDataModule, KGBatch
from kge.models.builder import KGEModel
from kge.trainer.base import BaseTrainer
from kge.utils.metrics import accuracy

logger = logging.getLogger(__name__)


class RelationPredictionTrainer(BaseTrainer):
    """关系预测训练器

    冻结 encoder，训练 RelationPredictionHead
    """

    def __init__(
        self,
        cfg: Config,
        model: KGEModel,
        data_module: KGDataModule,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, model, data_module, device)
        # 冻结 encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _train_step(self, batch: KGBatch) -> tuple[float, dict[str, float]]:
        h = batch.pos_h.to(self.device)
        r = batch.pos_r.to(self.device)
        t = batch.pos_t.to(self.device)

        logits = self.model(h, r, t)  # (B, num_relations)
        loss = F.cross_entropy(logits, r)
        acc = (logits.argmax(dim=-1) == r).float().mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {"loss": loss.item(), "acc": acc}

    @torch.no_grad()
    def _eval(self, split: str) -> tuple[float, dict[str, float]]:
        triples = (
            self.data_module.dataset.val_triples
            if split == "val"
            else self.data_module.dataset.test_triples
        )
        triples = triples.to(self.device)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

        logits = self.model(h, r, t)
        loss = F.cross_entropy(logits, r).item()
        acc = accuracy(logits, r)

        return loss, {"loss": loss, "acc": acc}

    @property
    def _monitor_mode(self) -> str:
        return "max"

    @property
    def _monitor_metric(self) -> str:
        return "val_acc"
