from __future__ import annotations

import torch
from torch import nn

from damascus.forge.heat import ConfidenceHead, quality_gate
from damascus.smithy.settings import RbmConfig


class BilinearBridge(nn.Module):
    def __init__(self, embed_dim: int, cross_dim: int) -> None:
        super().__init__()
        self.interact = nn.Linear(embed_dim, cross_dim)
        self.project = nn.Linear(cross_dim, embed_dim)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        product = source * target
        gated = torch.sigmoid(self.interact(product))
        return self.project(gated)


class CrossTokenAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        heads = num_heads if embed_dim % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

    def forward(self, h_motion: torch.Tensor, h_image: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([h_motion, h_image], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens)
        return attended.mean(dim=1)


class ConfidencePair(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.motion_head = ConfidenceHead(embed_dim)
        self.image_head = ConfidenceHead(embed_dim)

    def forward(
        self, h_motion: torch.Tensor, h_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.motion_head(h_motion), self.image_head(h_image)


class RegulatoryBridge(nn.Module):
    def __init__(self, cfg: RbmConfig, embed_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_confidence = cfg.use_confidence and cfg.fusion != "concat"
        self.use_gating = cfg.use_gating and cfg.fusion != "concat"
        if self.use_confidence:
            self.confidence = ConfidencePair(embed_dim)
        if cfg.fusion == "bilinear":
            self.bridge_mi = BilinearBridge(embed_dim, cfg.cross_dim)
            self.bridge_im = BilinearBridge(embed_dim, cfg.cross_dim)
            self.norm = nn.LayerNorm(embed_dim)
        elif cfg.fusion == "cross_attn":
            self.cross = CrossTokenAttention(embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
        elif cfg.fusion == "concat":
            self.merge = nn.Linear(2 * embed_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"unknown fusion: {cfg.fusion}")

    def forward(
        self, h_motion: torch.Tensor, h_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ones = torch.ones(h_motion.shape[0], 1, device=h_motion.device, dtype=h_motion.dtype)
        if self.use_confidence:
            c_motion, c_image = self.confidence(h_motion, h_image)
        else:
            c_motion, c_image = ones, ones

        if self.use_gating:
            g_motion = quality_gate(c_motion, h_motion)
            g_image = quality_gate(c_image, h_image)
        else:
            g_motion, g_image = h_motion, h_image

        if self.cfg.fusion == "bilinear":
            cross_mi = self.bridge_mi(g_motion, g_image)
            cross_im = self.bridge_im(g_image, g_motion)
            fused = self.norm(g_motion + g_image + cross_mi + cross_im)
        elif self.cfg.fusion == "cross_attn":
            fused = self.norm(g_motion + g_image + self.cross(g_motion, g_image))
        else:
            fused = self.norm(self.merge(torch.cat([g_motion, g_image], dim=-1)))

        return fused, c_motion, c_image
