from torch import nn
import torch


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x
