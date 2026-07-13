"""任务预测头"""

import torch
from torch import nn, Tensor


class NodeClassificationHead(nn.Module):
    """节点分类头"""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(self.dropout(embeddings))


class LinkPredictionHead(nn.Module):
    """链接预测头，双线性（Dot Product）"""

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        """z: [N, H]; edge_label_index: [2, E]; 返回 [E]"""
        src, dst = edge_label_index[0], edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)


class LinkPredictionMLPHead(nn.Module):
    """链接预测头，MLP 拼接"""

    def __init__(self, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index[0], edge_label_index[1]
        edge_feat = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)
