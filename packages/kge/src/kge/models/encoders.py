from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, TypeVar, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RelationEmbedding: TypeAlias = nn.Embedding | nn.ModuleDict


class BaseKGEEncoder(nn.Module, ABC):
    """KGE Encoder 基类

    职责：存储实体/关系嵌入，查表 + score 计算。
    子类需实现 score()
    """

    # 类级类型注解 — 供静态类型检查器与 builder.py 使用
    num_entities: int
    num_relations: int
    embedding_dim: int
    entity_embedding: nn.Embedding
    relation_embedding: RelationEmbedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_embedding = self._build_entity_embedding()
        self.relation_embedding = self._build_relation_embedding()
        self.reset_parameters()

    def _build_entity_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.num_entities, self.embedding_dim)

    def _build_relation_embedding(self) -> RelationEmbedding:
        """子类覆写以返回 ModuleDict (TransH/TransR/PairRE)"""
        return nn.Embedding(self.num_relations, self.embedding_dim)

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        r_emb = self.relation_embedding
        if isinstance(r_emb, nn.Embedding):
            nn.init.uniform_(r_emb.weight, -bound, bound)
        elif isinstance(r_emb, nn.ModuleDict):
            for v in r_emb.values():
                if isinstance(v, nn.Embedding):
                    nn.init.uniform_(v.weight, -bound, bound)
                else:
                    raise TypeError(
                        f"ModuleDict 中的值必须是 nn.Embedding，实际为 {type(v).__name__}"
                    )

    def _get_relation_embedding(self, r: Tensor) -> Tensor:
        """查关系嵌入。子类覆写以支持 ModuleDict"""
        relation_embedding = self.relation_embedding
        if not isinstance(relation_embedding, nn.Embedding):
            raise NotImplementedError
        return relation_embedding(r)

    def encode(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """查表返回嵌入"""
        h_emb = self.entity_embedding(h)
        r_emb = self._get_relation_embedding(r)
        t_emb = self.entity_embedding(t) if t is not None else None
        return h_emb, r_emb, t_emb

    @abstractmethod
    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        """计算三元组得分，越大越可信。

        注意：对于需要额外关系参数的模型（TransH 的 w、TransR 的 M_r、PairRE 的 rh/rt），
        score() 可能退化为简化计算。正确的评分应通过 forward() 获取。
        """
        ...

    def score_all_tails(self, head: Tensor, relation: Tensor) -> Tensor:
        """1-N 打分：(B, num_entities)"""
        h_emb, r_emb, _ = self.encode(head, relation, None)
        all_entities = self.entity_embedding.weight  # (E, d)

        if isinstance(self.relation_embedding, nn.ModuleDict):
            # ModuleDict 关系嵌入不支持广播，回退逐实体循环
            return self._score_all_tails_loop(h_emb, r_emb, head, relation)

        return self._score_all_tails_broadcast(h_emb, r_emb, all_entities)

    def _score_all_tails_broadcast(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        all_entities: Tensor,
    ) -> Tensor:
        """广播优化的 1-N 打分 (默认实现，适用于标准 Embedding)"""
        B, D = h_emb.shape
        E = all_entities.size(0)
        h_exp = h_emb.unsqueeze(1).expand(B, E, D)  # (B, E, d)
        r_exp = r_emb.unsqueeze(1).expand(B, E, D)
        t_exp = all_entities.unsqueeze(0).expand(B, E, D)
        return self.score(h_exp, r_exp, t_exp)  # (B, E)

    def _score_all_tails_loop(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        head: Tensor,
        relation: Tensor,
    ) -> Tensor:
        """逐实体循环 (慢但适用于 ModuleDict 关系嵌入)"""
        E = self.num_entities
        B = head.size(0)
        scores = torch.empty(B, E, device=h_emb.device)
        all_t = torch.arange(E, device=h_emb.device)
        for i in range(E):
            t_idx = all_t[i : i + 1].expand(B)
            _, _, t_emb = self.encode(head, relation, t_idx)
            assert t_emb is not None
            scores[:, i] = self.score(h_emb, r_emb, t_emb)
        return scores

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_emb, r_emb, t_emb = self.encode(h, r, t)
        assert t_emb is not None
        return self.score(h_emb, r_emb, t_emb)

    @staticmethod
    def _embedding_regularize(*embeddings: Tensor, p: int = 2) -> Tensor:
        """计算 sum(||x||_p^p)，返回标量 tensor"""
        if not embeddings:
            return torch.tensor(0.0)
        reg = torch.tensor(0.0, device=embeddings[0].device)
        for emb in embeddings:
            reg = reg + emb.norm(p=p).pow(p)
        return reg

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        """逐模型嵌入正则化，默认返回 0。子类覆写实现特定正则项"""
        return torch.tensor(0.0, device=self.entity_embedding.weight.device)


KGE_ENCODER_REGISTRY: dict[str, type[BaseKGEEncoder]] = {}
T = TypeVar("T", bound=type[BaseKGEEncoder])


def register(name: str) -> Callable[[T], T]:
    def dec(cls):
        KGE_ENCODER_REGISTRY[name] = cls
        return cls

    return dec


@register("trans-e")
class TransEEncoder(BaseKGEEncoder):
    """得分 = gamma - ||h + r - t||_p"""

    relation_embedding: nn.Embedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
        p_norm: int = 1,
        **kwargs: object,
    ) -> None:
        self.gamma = gamma
        self.p_norm = p_norm
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        F.normalize(self.entity_embedding.weight.data, p=2, dim=-1)
        nn.init.uniform_(self.relation_embedding.weight, -bound, bound)

    def score(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        return self.gamma - torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, r, t, p=2)


@register("trans-h")
class TransHEncoder(BaseKGEEncoder):
    """超平面平移，得分 = gamma - ||proj(h, w) + r - proj(t, w)||"""

    relation_embedding: nn.ModuleDict

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
        p_norm: int = 1,
        **kwargs: object,
    ) -> None:
        self.gamma = gamma
        self.p_norm = p_norm
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    def _build_relation_embedding(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "r": nn.Embedding(self.num_relations, self.embedding_dim),
                "w": nn.Embedding(self.num_relations, self.embedding_dim),
            }
        )

    def _get_relation_embedding(self, r: Tensor) -> Tensor:
        assert isinstance(self.relation_embedding, nn.ModuleDict)
        return cast(nn.Embedding, self.relation_embedding["r"])(r)

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(
            cast(nn.Embedding, self.relation_embedding["r"]).weight, -bound, bound
        )
        nn.init.uniform_(
            cast(nn.Embedding, self.relation_embedding["w"]).weight, -bound, bound
        )

    @staticmethod
    def _project(e: Tensor, w: Tensor) -> Tensor:
        dot = torch.sum(e * w, dim=-1, keepdim=True)
        return e - dot * w

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        """朴素得分函数，不使用超平面投影。

        由于 score_all_tails 广播计算无法获取逐关系超平面法向量 w，
        此处退化为简单范数 ‖h + r - t‖。正确实现见 forward()。

        See Also:
            forward(): 包含超平面投影 h_proj, t_proj 的正确 TransH 评分。
        """
        return self.gamma - torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_emb = self.entity_embedding(h)
        r_vec = cast(nn.Embedding, self.relation_embedding["r"])(r)
        w = F.normalize(
            cast(nn.Embedding, self.relation_embedding["w"])(r), p=2, dim=-1
        )
        t_emb = self.entity_embedding(t)
        h_proj = self._project(h_emb, w)
        t_proj = self._project(t_emb, w)
        return self.gamma - torch.norm(h_proj + r_vec - t_proj, p=self.p_norm, dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        r = cast(nn.Embedding, self.relation_embedding["r"])(relation)
        w = F.normalize(cast(nn.Embedding, self.relation_embedding["w"])(relation), p=2, dim=-1)
        h = self.entity_embedding(head)
        t = self.entity_embedding(tail)
        entity_reg = self._embedding_regularize(h, t, p=2) * 0.01
        rel_reg = self._embedding_regularize(r, p=2) * 0.01
        ortho_h = torch.sum(w * h, dim=-1).pow(2).mean()
        ortho_t = torch.sum(w * t, dim=-1).pow(2).mean()
        w_norm_reg = (torch.norm(w, p=2, dim=-1) - 1).pow(2).mean()
        return entity_reg + rel_reg + (ortho_h + ortho_t) * 0.01 + w_norm_reg


@register("trans-r")
class TransREncoder(BaseKGEEncoder):
    """关系空间平移，得分 = gamma - ||M_r(h) + r - M_r(t)||_p"""

    relation_embedding: nn.ModuleDict

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        relation_dim: int = 100,
        gamma: float = 12.0,
        p_norm: int = 1,
        **kwargs: object,
    ) -> None:
        self.relation_dim = relation_dim
        self.gamma = gamma
        self.p_norm = p_norm
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    def _build_relation_embedding(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "r": nn.Embedding(self.num_relations, self.relation_dim),
                "M": nn.Embedding(
                    self.num_relations, self.relation_dim * self.embedding_dim
                ),
            }
        )

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(cast(nn.Embedding, self.relation_embedding["r"]).weight, -bound, bound)
        # M 初始化为恒等矩阵（展平），稳定早期训练
        with torch.no_grad():
            I_flat = torch.eye(self.relation_dim, self.embedding_dim).flatten()
            cast(nn.Embedding, self.relation_embedding["M"]).weight.data.copy_(
                I_flat.unsqueeze(0).repeat(self.num_relations, 1)
            )

    def _get_relation_emb(self, r: Tensor) -> Tensor:
        return cast(nn.Embedding, self.relation_embedding["r"])(r)

    def _project(self, e: Tensor, M_flat: Tensor) -> Tensor:
        B = e.size(0)
        M = M_flat.view(-1, self.relation_dim, self.embedding_dim)
        return torch.bmm(M, e.unsqueeze(-1)).squeeze(-1)

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_emb = self.entity_embedding(h)
        t_emb = self.entity_embedding(t)
        r_vec = cast(nn.Embedding, self.relation_embedding["r"])(r)
        M = cast(nn.Embedding, self.relation_embedding["M"])(r)
        h_proj = self._project(h_emb, M)
        t_proj = self._project(t_emb, M)
        return self.gamma - torch.norm(h_proj + r_vec - t_proj, p=self.p_norm, dim=-1)

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        """朴素得分函数，不使用关系投影矩阵

        由于 score_all_tails 广播计算无法获取逐关系投影矩阵 M_r，
        此处退化为简单范数 ‖h + r - t‖。正确实现见 forward()

        See Also:
            forward(): 包含关系空间投影 h_proj = M_r(h), t_proj = M_r(t) 的正确 TransR 评分
        """
        return self.gamma - torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = cast(nn.Embedding, self.relation_embedding["r"])(relation)
        t = self.entity_embedding(tail)
        M_weight = cast(nn.Embedding, self.relation_embedding["M"])(relation)
        entity_reg = self._embedding_regularize(h, t, p=2) * 0.01
        rel_reg = self._embedding_regularize(r, p=2) * 0.01
        M_reg = self._embedding_regularize(M_weight, p=2)
        return entity_reg + rel_reg + M_reg


@register("dist-mult")
class DistMultEncoder(BaseKGEEncoder):
    """双线性对角模型，得分 = sum(h * r * t)"""

    relation_embedding: nn.Embedding

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        return torch.sum(h_emb * r_emb * t_emb, dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, r, t, p=3)


@register("compl-ex")
class ComplExEncoder(BaseKGEEncoder):
    """复数嵌入，得分 = Re(Σ h * r * conj(t))"""

    relation_embedding: nn.Embedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        **kwargs: object,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim * 2, **kwargs)

    @property
    def real_dim(self) -> int:
        return self.embedding_dim // 2

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        d = self.real_dim
        h_re, h_im = h_emb[..., :d], h_emb[..., d:]
        r_re, r_im = r_emb[..., :d], r_emb[..., d:]
        t_re, t_im = t_emb[..., :d], t_emb[..., d:]
        return torch.sum(
            (h_re * r_re - h_im * r_im) * t_re + (h_re * r_im + h_im * r_re) * t_im,
            dim=-1,
        )

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, r, t, p=3)


@register("rotat-e")
class RotatEEncoder(BaseKGEEncoder):
    """复数旋转，得分 = gamma - ||h ∘ r - t||   (r 在复平面单位圆上)"""

    relation_embedding: nn.Embedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
        epsilon: float = 2.0,
        **kwargs: object,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    def _build_entity_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.num_entities, self.embedding_dim * 2)

    def _build_relation_embedding(self) -> nn.Embedding:
        # 关系只存相位角 θ ∈ [0, 2π)
        return nn.Embedding(self.num_relations, self.embedding_dim)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.uniform_(self.relation_embedding.weight, 0, 2 * math.pi)

    @property
    def real_dim(self) -> int:
        return self.embedding_dim

    def _get_relation_emb(self, r: Tensor) -> Tensor:
        theta = self.relation_embedding(r)
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

    def _get_relation_embedding(self, r: Tensor) -> Tensor:
        """覆写基类：关系嵌入需从相位角 θ 展开为 (cos(θ), sin(θ))。"""
        return self._get_relation_emb(r)

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        d = self.real_dim
        h_re, h_im = h_emb[..., :d], h_emb[..., d:]
        t_re, t_im = t_emb[..., :d], t_emb[..., d:]
        r_re, r_im = r_emb[..., :d], r_emb[..., d:]

        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re

        return self.gamma - torch.sqrt(
            (h_rot_re - t_re) ** 2 + (h_rot_im - t_im) ** 2 + self.epsilon
        ).sum(dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, t, p=2)


@register("pair-re")
class PairREEncoder(BaseKGEEncoder):
    """成对关系向量，得分 = gamma - ||h * r^H - t * r^T||"""

    relation_embedding: nn.ModuleDict

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
        p_norm: int = 1,
        **kwargs: object,
    ) -> None:
        self.gamma = gamma
        self.p_norm = p_norm
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    def _build_relation_embedding(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "rh": nn.Embedding(self.num_relations, self.embedding_dim),
                "rt": nn.Embedding(self.num_relations, self.embedding_dim),
            }
        )

    def _get_relation_emb(self, r: Tensor) -> Tensor:
        return cast(nn.Embedding, self.relation_embedding["rh"])(r)

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.ones_(cast(nn.Embedding, self.relation_embedding["rh"]).weight)
        nn.init.ones_(cast(nn.Embedding, self.relation_embedding["rt"]).weight)

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_emb = self.entity_embedding(h)
        t_emb = self.entity_embedding(t)
        rh = cast(nn.Embedding, self.relation_embedding["rh"])(r)
        rt = cast(nn.Embedding, self.relation_embedding["rt"])(r)
        return self.gamma - torch.norm(h_emb * rh - t_emb * rt, p=self.p_norm, dim=-1)

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        """朴素得分函数，不使用成对关系向量

        由于 score_all_tails 广播计算无法获取逐关系 rh/rt 向量，
        此处退化为简单范数 ‖h - t‖。正确实现见 forward()

        See Also:
            forward(): 包含 rh, rt 逐元素乘法的正确 PairRE 评分
        """
        return self.gamma - torch.norm(h_emb - t_emb, p=self.p_norm, dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        rh = cast(nn.Embedding, self.relation_embedding["rh"])(relation)
        rt = cast(nn.Embedding, self.relation_embedding["rt"])(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, rh, rt, t, p=2)


@register("conv-e")
class ConvEEncoder(BaseKGEEncoder):
    """2D 卷积 + FC + 与尾实体内积"""

    relation_embedding: nn.Embedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        conv_out_channels: int = 32,
        kernel_size: int = 3,
        input_dropout: float = 0.2,
        feature_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_height: int = 10,
        embedding_width: int = 20,
        **kwargs: object,
    ) -> None:
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.input_dropout = input_dropout
        self.feature_dropout = feature_dropout
        self.hidden_dropout = hidden_dropout
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width
        # embedding_dim = height * width
        actual_dim = embedding_height * embedding_width
        super().__init__(num_entities, num_relations, actual_dim, **kwargs)
        self._conv_layer: nn.Module | None = None
        self._fc: nn.Module | None = None

    def _build_relation_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.num_relations, self.embedding_dim)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def _build_conv_layers(self) -> nn.Module:
        h, w = self.embedding_height, self.embedding_width
        return nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.input_dropout),
            nn.Conv2d(1, self.conv_out_channels, (self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.conv_out_channels),
            nn.Dropout(self.feature_dropout),
            nn.ReLU(),
        )

    def _build_fc(self) -> nn.Module:
        h, w = self.embedding_height, self.embedding_width
        conv_out_h = 2 * h - self.kernel_size + 1
        conv_out_w = w - self.kernel_size + 1
        flat_dim = self.conv_out_channels * conv_out_h * conv_out_w
        return nn.Linear(flat_dim, self.embedding_dim)

    def _ensure_conv_layers(self, device: torch.device) -> None:
        """延迟构建卷积层和全连接层（首次调用时初始化到目标设备）。"""
        if self._conv_layer is None:
            self._conv_layer = self._build_conv_layers().to(device)
            self._fc = self._build_fc().to(device)

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        B = h_emb.size(0)
        h, w = self.embedding_height, self.embedding_width
        x = torch.cat([h_emb, r_emb], dim=-1).view(B, 1, 2 * h, w)

        self._ensure_conv_layers(h_emb.device)
        assert self._conv_layer is not None and self._fc is not None
        x = self._conv_layer(x).view(B, -1)
        x = F.dropout(
            F.relu(self._fc(x)),
            p=self.hidden_dropout,
            training=self.training,
        )
        return torch.sum(x * t_emb, dim=-1)

    def score_all_tails(self, head: Tensor, relation: Tensor) -> Tensor:
        h_emb = self.entity_embedding(head)
        r_emb = self.relation_embedding(relation)
        B = h_emb.size(0)
        h, w = self.embedding_height, self.embedding_width
        x = torch.cat([h_emb, r_emb], dim=-1).view(B, 1, 2 * h, w)

        self._ensure_conv_layers(h_emb.device)
        assert self._conv_layer is not None and self._fc is not None
        x = self._conv_layer(x).view(B, -1)
        x = F.dropout(
            F.relu(self._fc(x)),
            p=self.hidden_dropout,
            training=self.training,
        )
        # 矩阵乘法：(B, d) @ (d, E) → (B, E)
        return x @ self.entity_embedding.weight.T

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, r, t, p=2) * 0.01


Quaternion: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


@register("quat-e")
class QuatEEncoder(BaseKGEEncoder):
    """四元数嵌入，得分 = (h ⊗ r^◁) · t    (Hamilton 积 + 四元数内积)"""

    relation_embedding: nn.Embedding

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        **kwargs: object,
    ) -> None:
        # embedding_dim 必须是 4 的倍数
        assert embedding_dim % 4 == 0, "QuatE 嵌入维度必须是 4 的倍数"
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

    @property
    def quat_dim(self) -> int:
        return self.embedding_dim // 4

    @staticmethod
    def _hamilton_product(
        a1: Tensor,
        b1: Tensor,
        c1: Tensor,
        d1: Tensor,
        a2: Tensor,
        b2: Tensor,
        c2: Tensor,
        d2: Tensor,
    ) -> Quaternion:
        return (
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        )

    def _normalize_quat(self, q_parts: Quaternion) -> Quaternion:
        a, b, c, d = q_parts
        norm = torch.sqrt(a**2 + b**2 + c**2 + d**2 + 1e-8)
        return a / norm, b / norm, c / norm, d / norm

    def _split(self, x: Tensor) -> Quaternion:
        d = self.quat_dim
        return x[..., :d], x[..., d : 2 * d], x[..., 2 * d : 3 * d], x[..., 3 * d :]

    def score(self, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor) -> Tensor:
        ha, hb, hc, hd = self._split(h_emb)
        ra, rb, rc, rd = self._normalize_quat(self._split(r_emb))
        ta, tb, tc, td = self._split(t_emb)

        ha_rot, hb_rot, hc_rot, hd_rot = self._hamilton_product(
            ha, hb, hc, hd, ra, rb, rc, rd
        )

        return (ha_rot * ta + hb_rot * tb + hc_rot * tc + hd_rot * td).sum(dim=-1)

    def regularize(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t = self.entity_embedding(tail)
        return self._embedding_regularize(h, r, t, p=2)
