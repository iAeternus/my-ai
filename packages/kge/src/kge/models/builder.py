from __future__ import annotations
from typing import Any
import torch.nn as nn
from torch import Tensor

from kge.config.schema import Config, TaskType
from kge.models.encoders import BaseKGEEncoder, KGE_ENCODER_REGISTRY
from kge.models.heads import BaseHead, HEAD_REGISTRY
from core.utils import dict_get_or_default


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
        encoder_cls = KGE_ENCODER_REGISTRY[encoder_name]
    except KeyError:
        available = sorted(KGE_ENCODER_REGISTRY.keys())
        raise KeyError(f"未知 encoder: {encoder_name!r}。可用: {available}") from None

    encoder_params: dict[str, Any] = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "embedding_dim": dict_get_or_default(p, "embedding_dim", 100),
    }

    match encoder_name:
        case "trans-e" | "trans-h":
            encoder_params["gamma"] = dict_get_or_default(p, "gamma", 12.0)
            encoder_params["p_norm"] = dict_get_or_default(p, "p_norm", 1)
        case "trans-r":
            encoder_params["gamma"] = dict_get_or_default(p, "gamma", 12.0)
            encoder_params["p_norm"] = dict_get_or_default(p, "p_norm", 1)
            encoder_params["relation_dim"] = dict_get_or_default(p, "relation_dim", 100)
        case "rotat-e":
            encoder_params["gamma"] = dict_get_or_default(p, "gamma", 12.0)
            encoder_params["epsilon"] = dict_get_or_default(p, "epsilon", 2.0)
        case "pair-re":
            encoder_params["gamma"] = dict_get_or_default(p, "gamma", 12.0)
            encoder_params["p_norm"] = dict_get_or_default(p, "p_norm", 1)
        case "conv-e":
            encoder_params["conv_out_channels"] = dict_get_or_default(
                p, "conv_out_channels", 32
            )
            encoder_params["kernel_size"] = dict_get_or_default(p, "kernel_size", 3)
            encoder_params["input_dropout"] = dict_get_or_default(
                p, "input_dropout", 0.2
            )
            encoder_params["feature_dropout"] = dict_get_or_default(
                p, "feature_dropout", 0.2
            )
            encoder_params["hidden_dropout"] = dict_get_or_default(
                p, "hidden_dropout", 0.3
            )
        case _:
            pass  # 无需额外参数（dist-mult, compl-ex, quat-e）

    encoder = encoder_cls(**encoder_params)

    # Head
    head_name = cfg.model.head_name
    try:
        head_cls = HEAD_REGISTRY[head_name]
    except KeyError:
        available = sorted(HEAD_REGISTRY.keys())
        raise KeyError(f"未知 head: {head_name!r}。可用: {available}") from None

    head_params: dict[str, Any] = {}
    match head_name:
        case "link_prediction":
            head_params["encoder"] = encoder
        case "relation_prediction":
            head_params["embedding_dim"] = encoder.embedding_dim
            head_params["num_relations"] = num_relations
            head_params["hidden_dim"] = dict_get_or_default(p, "hidden_dim", 256)
            head_params["dropout"] = dict_get_or_default(p, "hidden_dropout", 0.3)
        case "triple_classification":
            head_params["embedding_dim"] = encoder.embedding_dim
            head_params["hidden_dim"] = dict_get_or_default(p, "hidden_dim", 256)
            head_params["dropout"] = dict_get_or_default(p, "hidden_dropout", 0.3)
        case _:
            pass

    head = head_cls(**head_params)

    return KGEModel(encoder, head, cfg.task)
