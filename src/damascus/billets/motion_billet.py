from __future__ import annotations

import math

import torch
from torch import nn

from damascus.smithy.settings import TmeConfig


class DilatedCausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(x, (self.pad, 0))
        out = self.act(self.norm(self.conv(padded)))
        return out + x


class TemporalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.query(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = self.drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, -1)
        return self.out(context)


def _positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return encoding


class TemporalMotionEncoder(nn.Module):
    def __init__(self, cfg: TmeConfig, embed_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_features, cfg.model_dim)
        self.variant = cfg.variant
        if cfg.variant == "gru":
            self.gru = nn.GRU(
                cfg.model_dim,
                cfg.model_dim,
                num_layers=2,
                batch_first=True,
                dropout=cfg.dropout,
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    DilatedCausalBlock(cfg.model_dim, cfg.kernel_size, 2**layer)
                    for layer in range(cfg.num_layers)
                ]
            )
            self.attention = TemporalSelfAttention(cfg.model_dim, cfg.num_heads, cfg.dropout)
            self.attn_norm = nn.LayerNorm(cfg.model_dim)
        self.output_proj = nn.Linear(cfg.model_dim, embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(motion)
        if self.cfg.positional and self.variant != "gru":
            z = z + _positional_encoding(z.shape[1], z.shape[2], z.device).unsqueeze(0)
        if self.variant == "gru":
            sequence, _ = self.gru(z)
        else:
            conv_in = z.transpose(1, 2)
            for block in self.blocks:
                conv_in = block(conv_in)
            sequence = conv_in.transpose(1, 2)
            sequence = self.attn_norm(sequence + self.attention(sequence))
        pooled = sequence.mean(dim=1)
        return self.output_proj(self.drop(pooled))
