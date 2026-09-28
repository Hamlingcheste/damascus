from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from damascus.billets.imaging_billet import ImagingFeatureEncoder
from damascus.billets.motion_billet import TemporalMotionEncoder
from damascus.forge.bevel import AuxiliaryEdge, ClassificationEdge
from damascus.forge.weld import RegulatoryBridge
from damascus.smithy.settings import DamascusConfig, ModelConfig


class RegBridgeOutput(NamedTuple):
    logits: torch.Tensor
    aux_motion: torch.Tensor
    aux_image: torch.Tensor
    confidence_motion: torch.Tensor
    confidence_image: torch.Tensor
    fused: torch.Tensor


class RegBridge(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.motion_encoder = TemporalMotionEncoder(cfg.tme, cfg.embed_dim)
        self.image_encoder = ImagingFeatureEncoder(cfg.ife, cfg.embed_dim)
        self.bridge = RegulatoryBridge(cfg.rbm, cfg.embed_dim)
        self.classifier = ClassificationEdge(cfg.head, cfg.embed_dim, cfg.num_classes)
        self.aux_motion = AuxiliaryEdge(cfg.embed_dim, cfg.num_classes)
        self.aux_image = AuxiliaryEdge(cfg.embed_dim, cfg.num_classes)

    def forward(self, motion: torch.Tensor, image: torch.Tensor) -> RegBridgeOutput:
        h_motion = self.motion_encoder(motion)
        h_image = self.image_encoder(image)
        fused, c_motion, c_image = self.bridge(h_motion, h_image)
        logits = self.classifier(fused)
        return RegBridgeOutput(
            logits=logits,
            aux_motion=self.aux_motion(h_motion),
            aux_image=self.aux_image(h_image),
            confidence_motion=c_motion,
            confidence_image=c_image,
            fused=fused,
        )


def build_regbridge(config: DamascusConfig) -> RegBridge:
    return RegBridge(config.model)


__all__ = ["DamascusConfig", "RegBridge", "RegBridgeOutput", "build_regbridge"]
