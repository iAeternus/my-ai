from __future__ import annotations
from abc import ABC, abstractmethod
from kge.models.encoders import BaseKGEEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseHead(nn.Module, ABC):
    """预测头基类"""

    @abstractmethod
    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        **kwargs,
    ) -> Tensor: ...


class LinkPredictionHead(BaseHead):
    """链接预测头，委托 encoder.score()"""

    def __init__(self, encoder: BaseKGEEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        **kwargs,
    ) -> Tensor:
        return self.encoder.score(h_emb, r_emb, t_emb)


class RelationPredictionHead(BaseHead):
    """关系预测头，concat(h, t) -> MLP -> num_relations"""

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        **kwargs,
    ) -> Tensor:
        x = torch.cat([h_emb, t_emb], dim=-1)
        return self.mlp(x)


class TripleClassificationHead(BaseHead):
    """三元组分类头，concat(h, r, t) -> MLP -> 2"""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        **kwargs,
    ) -> Tensor:
        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        return self.mlp(x)


class EntitySimilarityHead(BaseHead):
    """实体相似度头 — 直接返回实体嵌入，外部计算 cos-sim"""

    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        **kwargs,
    ) -> Tensor:
        return h_emb


# 注册字典
HEAD_REGISTRY: dict[str, type[BaseHead]] = {
    "link_prediction": LinkPredictionHead,
    "relation_prediction": RelationPredictionHead,
    "triple_classification": TripleClassificationHead,
    "entity_similarity": EntitySimilarityHead,
}
