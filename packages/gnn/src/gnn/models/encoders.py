"""编码器，负责图节点表示学习

支持 BatchNorm / LayerNorm 切换，以及 DropEdge 结构正则化。
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.utils import dropout_edge

from core import Registry

ENCODER_REGISTRY = Registry[type[nn.Module]]("gnn encoder", base_class=nn.Module)


def _make_norm(hidden_dim: int, norm_type: str) -> nn.Module:
    """创建归一化层"""
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    return nn.BatchNorm1d(hidden_dim)


@ENCODER_REGISTRY.register("gcn")
class GCNEncoder(nn.Module):
    """多层 GCNConv + Norm + ReLU + Dropout"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: str = "batch",
        dropedge: float = 0.0,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.norms.append(_make_norm(hidden_dim, norm))
        self.dropout = dropout
        self.dropedge = dropedge

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            ei = edge_index
            if self.dropedge > 0 and self.training:
                ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        ei = edge_index
        if self.dropedge > 0 and self.training:
            ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
        x = self.convs[-1](x, ei)
        x = self.norms[-1](x)
        x = F.relu(x)
        return x


@ENCODER_REGISTRY.register("gat")
class GATEncoder(nn.Module):
    """多头注意力，最后一层单头输出 hidden_dim"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
        norm: str = "batch",
        dropedge: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            in_ch = in_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_ch, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(_make_norm(hidden_dim * heads, norm))
        # 最后一层，单头输出
        in_last = in_dim if num_layers == 1 else hidden_dim * heads
        self.convs.append(GATConv(in_last, hidden_dim, heads=1, dropout=dropout))
        self.norms.append(_make_norm(hidden_dim, norm))
        self.dropout = dropout
        self.dropedge = dropedge

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            ei = edge_index
            if self.dropedge > 0 and self.training:
                ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        ei = edge_index
        if self.dropedge > 0 and self.training:
            ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
        x = self.convs[-1](x, ei)
        x = self.norms[-1](x)
        x = F.relu(x)
        return x


@ENCODER_REGISTRY.register("gin")
class GINEncoder(nn.Module):
    """GINConv(MLP) + Norm

    最后一层不应用 dropout，与 GCN/GAT/SAGE 保持一致
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: str = "batch",
        dropedge: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(_make_norm(hidden_dim, norm))
        self.dropout = dropout
        self.dropedge = dropedge

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            ei = edge_index
            if self.dropedge > 0 and self.training:
                ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层：不应用 dropout
        ei = edge_index
        if self.dropedge > 0 and self.training:
            ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
        x = self.convs[-1](x, ei)
        x = self.norms[-1](x)
        x = F.relu(x)
        return x


@ENCODER_REGISTRY.register("sage")
class SAGEEncoder(nn.Module):
    """SAGEConv + Norm + ReLU + Dropout"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
        norm: str = "batch",
        dropedge: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_ch, hidden_dim, aggr=aggr))
            self.norms.append(_make_norm(hidden_dim, norm))
        self.dropout = dropout
        self.dropedge = dropedge

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            ei = edge_index
            if self.dropedge > 0 and self.training:
                ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        ei = edge_index
        if self.dropedge > 0 and self.training:
            ei, _ = dropout_edge(edge_index, p=self.dropedge, training=True)
        x = self.convs[-1](x, ei)
        x = self.norms[-1](x)
        x = F.relu(x)
        return x


