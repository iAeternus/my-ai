"""LinkPredictionTrainer — 链接预测训练器"""
from __future__ import annotations
import logging

import torch
from torch import Tensor

from kge.config.schema import Config, LossType
from kge.datasets.data_module import KGDataModule, KGBatch
from kge.models.builder import KGEModel
from kge.trainer.base import BaseTrainer
from kge.trainer.losses import (
    adversarial_margin_loss,
    bce_loss,
    cross_entropy_1n,
    margin_ranking_loss,
)
from kge.utils.metrics import hits_at_k, mrr, ranks

logger = logging.getLogger(__name__)


class LinkPredictionTrainer(BaseTrainer):
    """链接预测训练器

    支持 margin_ranking / bce / cross_entropy 三种损失模式
    """

    def _train_step(self, batch: KGBatch) -> tuple[float, dict[str, float]]:
        h = batch.pos_h.to(self.device)
        r = batch.pos_r.to(self.device)
        t = batch.pos_t.to(self.device)
        loss_type = self.cfg.train.loss_type
        B = h.size(0)

        if loss_type in (LossType.MARGIN_RANKING, LossType.BCE):
            neg_h = batch.neg_h.to(self.device)
            neg_r = batch.neg_r.to(self.device)
            neg_t = batch.neg_t.to(self.device)
            K = self.cfg.dataset.num_negative_samples

            pos_scores = self.model.encoder.forward(h, r, t)  # (B,)
            neg_scores = self.model.encoder.forward(neg_h, neg_r, neg_t)  # (B*K,)
            neg_scores = neg_scores.view(B, K)

            if loss_type == LossType.MARGIN_RANKING:
                if self.cfg.train.adversarial_temperature > 0:
                    loss = adversarial_margin_loss(
                        pos_scores, neg_scores,
                        margin=self.cfg.train.margin,
                        temperature=self.cfg.train.adversarial_temperature,
                    )
                else:
                    loss = margin_ranking_loss(
                        pos_scores, neg_scores, margin=self.cfg.train.margin,
                    )
            else:
                loss = bce_loss(
                    pos_scores, neg_scores,
                    label_smoothing=self.cfg.train.label_smoothing,
                )

        elif loss_type == LossType.CROSS_ENTROPY:
            scores = self.model.encoder.score_all_tails(h, r)  # (B, num_entities)
            loss = cross_entropy_1n(
                scores, t, label_smoothing=self.cfg.train.label_smoothing,
            )
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

        # 正则化
        if self.cfg.train.regularization_weight > 0:
            loss = loss + self.cfg.train.regularization_weight * self.model.encoder.regularize(h, r, t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {"loss": loss.item()}

    @torch.no_grad()
    def _eval(self, split: str) -> tuple[float, dict[str, float]]:
        triples = (
            self.data_module.dataset.val_triples
            if split == "val"
            else self.data_module.dataset.test_triples
        )
        triples = triples.to(self.device)
        bs = self.cfg.dataset.batch_size * 4  # eval 时可用更大 batch
        filter_set = self.data_module.filter_set

        all_ranks: list[Tensor] = []

        for i in range(0, len(triples), bs):
            batch = triples[i : i + bs]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # 尾实体预测
            scores_tail = self.model.encoder.score_all_tails(h, r)
            rank_tail = ranks(scores_tail, t, filter_set, h, r)
            all_ranks.append(rank_tail)

            # 头实体预测 (交换 h/t)
            scores_head = self.model.encoder.score_all_tails(t, r)
            rank_head = ranks(scores_head, h, filter_set, t, r)
            all_ranks.append(rank_head)

        all_ranks_tensor = torch.cat(all_ranks)
        metrics: dict[str, float] = {
            "mrr": mrr(all_ranks_tensor),
        }
        for k, v in hits_at_k(all_ranks_tensor, self.cfg.train.eval_ks).items():
            metrics[f"hits@{k}"] = v

        return 0.0, metrics

    @property
    def _monitor_mode(self) -> str:
        return "max"

    @property
    def _monitor_metric(self) -> str:
        return "val_mrr"
