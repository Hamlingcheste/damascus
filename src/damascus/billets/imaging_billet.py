from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from damascus.smithy.settings import IfeConfig


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = feat.mean(dim=(2, 3))
        weight = torch.sigmoid(self.fc2(self.act(self.fc1(pooled))))
        return feat * weight.unsqueeze(-1).unsqueeze(-1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        avg = feat.mean(dim=1, keepdim=True)
        mx = feat.max(dim=1, keepdim=True).values
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return feat * weight


def _build_backbone(name: str, in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    factory = getattr(models, name)
    weights = "DEFAULT" if pretrained else None
    try:
        net = factory(weights=weights)
    except Exception:
        net = factory(weights=None)
    channels = int(net.classifier.in_features)
    features = net.features
    if in_channels != 3:
        first = features.conv0
        replacement = nn.Conv2d(
            in_channels,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=first.bias is not None,
        )
        features.conv0 = replacement
    return features, channels


class ImagingFeatureEncoder(nn.Module):
    def __init__(self, cfg: IfeConfig, embed_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.features, channels = _build_backbone(cfg.backbone, cfg.in_channels, cfg.pretrained)
        self.use_attention = cfg.use_attention
        if cfg.use_attention:
            self.channel_gate = ChannelGate(channels, cfg.attn_reduction)
            self.spatial_gate = SpatialGate(cfg.spatial_kernel)
        self.output_proj = nn.Linear(channels, embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = torch.relu(self.features(image))
        if self.use_attention:
            feat = self.channel_gate(feat)
            feat = self.spatial_gate(feat)
        pooled = feat.mean(dim=(2, 3))
        return self.output_proj(self.drop(pooled))
