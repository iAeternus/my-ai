"""create_trainer — 按 TaskType 分发"""
from __future__ import annotations

import torch

from kge.config.schema import Config, TaskType
from kge.datasets.data_module import KGDataModule
from kge.models.builder import KGEModel
from kge.trainer.base import BaseTrainer
from kge.trainer.link_prediction import LinkPredictionTrainer
from kge.trainer.relation_prediction import RelationPredictionTrainer
from kge.trainer.triple_classification import TripleClassificationTrainer


def create_trainer(
    cfg: Config,
    model: KGEModel,
    data_module: KGDataModule,
    device: torch.device,
) -> BaseTrainer:
    """根据 TaskType 创建对应的训练器"""
    match cfg.task:
        case TaskType.LINK_PREDICTION:
            return LinkPredictionTrainer(cfg, model, data_module, device)
        case TaskType.RELATION_PREDICTION:
            return RelationPredictionTrainer(cfg, model, data_module, device)
        case TaskType.TRIPLE_CLASSIFICATION:
            return TripleClassificationTrainer(cfg, model, data_module, device)
        case _:
            raise ValueError(f"未知任务类型: {cfg.task}")
