"""编码器，负责图节点表示学习"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv


class GCNEncoder(nn.Module):
    """多层 GCNConv + BatchNorm + ReLU + Dropout"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.relu(x)
        return x


class GATEncoder(nn.Module):
    """多头注意力，最后一层单头输出 hidden_dim"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):
            in_ch = in_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_ch, hidden_dim, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        # 最后一层，单头输出
        in_last = in_dim if num_layers == 1 else hidden_dim * heads
        self.convs.append(GATConv(in_last, hidden_dim, heads=1, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.relu(x)
        return x


class GINEncoder(nn.Module):
    """GINConv(MLP) + BatchNorm"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGEEncoder(nn.Module):
    """SAGEConv + BatchNorm + ReLU + Dropout"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_ch, hidden_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.relu(x)
        return x


ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "gcn": GCNEncoder,
    "gat": GATEncoder,
    "gin": GINEncoder,
    "sage": SAGEEncoder,
}
