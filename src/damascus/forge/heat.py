from __future__ import annotations

import torch
from torch import nn


class ConfidenceHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = max(1, embed_dim // 4)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        score = self.fc2(self.act(self.fc1(embedding)))
        return torch.sigmoid(score)


def quality_gate(confidence: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    return confidence * embedding
