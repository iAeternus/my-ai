from __future__ import annotations
import logging

import torch
import torch.nn.functional as F

from kge.config.schema import Config
from kge.datasets.data_module import KGDataModule, KGBatch
from kge.datasets.sampler import UniformNegativeSampler
from kge.models.builder import KGEModel
from kge.trainer.base import BaseTrainer
from kge.utils.metrics import accuracy

logger = logging.getLogger(__name__)


class TripleClassificationTrainer(BaseTrainer):
    """三元组分类训练器

    冻结 encoder，训练 TripleClassificationHead
    每正例采样 1 个负例进行二分类
    """

    def __init__(
        self,
        cfg: Config,
        model: KGEModel,
        data_module: KGDataModule,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, model, data_module, device)
        self.neg_sampler = UniformNegativeSampler(data_module.num_entities)
        # 冻结 encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _train_step(self, batch: KGBatch) -> tuple[float, dict[str, float]]:
        h = batch.pos_h.to(self.device)
        r = batch.pos_r.to(self.device)
        t = batch.pos_t.to(self.device)
        B = h.size(0)

        # 每正例采样 1 个负例
        neg_h, neg_r, neg_t = self.neg_sampler.sample(
            h, r, t, num_neg=1, device=self.device
        )

        pos_logits = self.model(h, r, t)  # (B, 2)
        neg_logits = self.model(neg_h, neg_r, neg_t)  # (B, 2)
        logits = torch.cat([pos_logits, neg_logits], dim=0)  # (2B, 2)
        labels = torch.cat(
            [
                torch.ones(B, dtype=torch.long, device=self.device),
                torch.zeros(B, dtype=torch.long, device=self.device),
            ]
        )

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()

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
        B = len(triples)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

        neg_h, neg_r, neg_t = self.neg_sampler.sample(
            h, r, t, num_neg=1, device=self.device
        )

        pos_logits = self.model(h, r, t)
        neg_logits = self.model(neg_h, neg_r, neg_t)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(B, dtype=torch.long, device=self.device),
                torch.zeros(B, dtype=torch.long, device=self.device),
            ]
        )

        loss = F.cross_entropy(logits, labels).item()
        acc = accuracy(logits, labels)

        return loss, {"loss": loss, "acc": acc}

    @property
    def _monitor_mode(self) -> str:
        return "max"

    @property
    def _monitor_metric(self) -> str:
        return "val_acc"
