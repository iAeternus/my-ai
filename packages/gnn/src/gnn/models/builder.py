"""模型组装工厂"""

from core.utils import dict_get_or_default
import torch
from torch import nn, Tensor
from gnn.config.schema import Config, TaskType
from gnn.models.encoders import ENCODER_REGISTRY
from gnn.models.heads import (
    NodeClassificationHead,
    LinkPredictionHead,
    LinkPredictionMLPHead,
)


class GNNModel(nn.Module):
    """组合encoder和head的完整模型"""

    def __init__(self, encoder: nn.Module, head: nn.Module, task: TaskType) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.task = task

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        z = self.encoder(x, edge_index)
        if self.task == TaskType.NODE_CLASSIFICATION:
            return self.head(z)
        elif self.task == TaskType.LINK_PREDICTION:
            return self.head(z, kwargs["edge_label_index"])
        else:
            raise ValueError(f"Unknown task: {self.task}")


def build_model(cfg: Config, num_features: int, num_classes: int) -> GNNModel:
    # encoder
    encoder_cls = ENCODER_REGISTRY[cfg.model.name]
    encoder_params = {
        "in_dim": num_features,
        "hidden_dim": dict_get_or_default(cfg.model.params, "hidden_dim", 64),
        "num_layers": dict_get_or_default(cfg.model.params, "num_layers", 2),
        "dropout": dict_get_or_default(cfg.model.params, "dropout", 0.5),
        "norm": dict_get_or_default(cfg.model.params, "norm", "batch"),
        "dropedge": dict_get_or_default(cfg.model.params, "dropedge", 0.0),
    }
    if cfg.model.name == "gat":
        encoder_params["heads"] = dict_get_or_default(cfg.model.params, "heads", 8)
    if cfg.model.name == "sage":
        encoder_params["aggr"] = dict_get_or_default(cfg.model.params, "aggr", "mean")

    encoder = encoder_cls(**encoder_params)

    # head
    hidden_dim = dict_get_or_default(cfg.model.params, "hidden_dim", 64)
    if cfg.task == TaskType.NODE_CLASSIFICATION:
        head = NodeClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dict_get_or_default(cfg.model.params, "dropout", 0.5),
        )
    elif cfg.task == TaskType.LINK_PREDICTION:
        predictor_type = dict_get_or_default(cfg.model.params, "link_predictor", "dot_product")
        if predictor_type == "mlp":
            head = LinkPredictionMLPHead(
                hidden_dim=hidden_dim,
                dropout=dict_get_or_default(cfg.model.params, "dropout", 0.5),
            )
        else:
            head = LinkPredictionHead()
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    return GNNModel(encoder, head, cfg.task)
