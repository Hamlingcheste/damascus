from __future__ import annotations

import torch
from torch import nn

from damascus.smithy.settings import HeadConfig


class ClassificationEdge(nn.Module):
    def __init__(self, cfg: HeadConfig, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.hidden_dim, num_classes)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(fused))))


class AuxiliaryEdge(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.proj(embedding)
