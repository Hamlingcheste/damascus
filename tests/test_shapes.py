from __future__ import annotations

import torch

from conftest import tiny_data_config, tiny_model_config
from damascus.billets.imaging_billet import ImagingFeatureEncoder
from damascus.billets.motion_billet import TemporalMotionEncoder
from damascus.blade import RegBridge
from damascus.stock.cohorts import forge_split


def test_encoder_output_dims() -> None:
    cfg = tiny_model_config()
    motion = torch.randn(5, cfg.tme.seq_len, cfg.tme.in_features)
    image = torch.randn(5, cfg.ife.in_channels, 64, 64)
    motion_enc = TemporalMotionEncoder(cfg.tme, cfg.embed_dim)
    image_enc = ImagingFeatureEncoder(cfg.ife, cfg.embed_dim)
    assert motion_enc(motion).shape == (5, cfg.embed_dim)
    assert image_enc(image).shape == (5, cfg.embed_dim)


def test_model_forward_shapes() -> None:
    mcfg = tiny_model_config()
    dcfg = tiny_data_config()
    model = RegBridge(mcfg)
    split = forge_split(dcfg, 6, 0)
    out = model(split["motion"], split["image"])
    assert out.logits.shape == (6, mcfg.num_classes)
    assert out.aux_motion.shape == (6, mcfg.num_classes)
    assert out.confidence_motion.shape == (6, 1)
    assert out.fused.shape == (6, mcfg.embed_dim)


def test_confidence_in_unit_interval() -> None:
    mcfg = tiny_model_config()
    model = RegBridge(mcfg)
    split = forge_split(tiny_data_config(), 8, 1)
    out = model(split["motion"], split["image"])
    for tensor in (out.confidence_motion, out.confidence_image):
        assert torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0)
