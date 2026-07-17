from __future__ import annotations
from typing import Any
import torch.nn as nn
from torch import Tensor

from kge.config.schema import Config, TaskType
from kge.models.encoders import BaseKGEEncoder, ENCODER_REGISTRY
from kge.models.heads import BaseHead, HEAD_REGISTRY


class KGEModel(nn.Module):
    """组合 encoder + head 的完整 KGE 模型"""

    def __init__(
        self,
        encoder: BaseKGEEncoder,
        head: BaseHead,
        task: TaskType,
    ) -> None:
        super().__init__()
        self.encoder: BaseKGEEncoder = encoder
        self.head: BaseHead = head
        self.task: TaskType = task

    def forward(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        h_emb, r_emb, t_emb = self.encoder.encode(h, r, t)
        return self.head(h_emb, r_emb, t_emb, **kwargs)


def build_model(cfg: Config, num_entities: int, num_relations: int) -> KGEModel:
    """从 Config 构建 KGEModel

    根据 encoder_name / head_name 从注册表查找对应类，
    提取所需参数后实例化
    """
    p = cfg.model.params

    # Encoder
    encoder_name = cfg.model.encoder_name
    try:
        encoder_cls = ENCODER_REGISTRY[encoder_name]
    except KeyError:
        available = sorted(ENCODER_REGISTRY.keys())
        raise KeyError(f"未知 encoder: {encoder_name!r}。可用: {available}") from None

    encoder = encoder_cls.from_config(
        cfg,
        num_entities=num_entities,
        num_relations=num_relations,
    )

    # Head
    head_name = cfg.model.head_name
    try:
        head_cls = HEAD_REGISTRY[head_name]
    except KeyError:
        available = sorted(HEAD_REGISTRY.keys())
        raise KeyError(f"未知 head: {head_name!r}。可用: {available}") from None

    head = head_cls.from_config(
        cfg,
        encoder=encoder,
        num_relations=num_relations,
    )

    return KGEModel(encoder, head, cfg.task)
