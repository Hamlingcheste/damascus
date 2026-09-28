from __future__ import annotations

import torch

from conftest import tiny_data_config, tiny_model_config
from damascus.blade import RegBridge
from damascus.smithy.settings import IfeConfig, ModelConfig, RbmConfig, TmeConfig
from damascus.stock.cohorts import forge_split


def _run(model: RegBridge) -> tuple[torch.Tensor, torch.Tensor]:
    split = forge_split(tiny_data_config(), 6, 0)
    out = model(split["motion"], split["image"])
    return out.logits, out.confidence_motion


def test_each_fusion_variant_produces_logits() -> None:
    for fusion in ("bilinear", "cross_attn", "concat"):
        model = RegBridge(tiny_model_config(fusion=fusion))
        logits, _ = _run(model)
        assert logits.shape == (6, 4)


def test_concat_variant_disables_confidence() -> None:
    model = RegBridge(tiny_model_config(fusion="concat"))
    _, confidence = _run(model)
    assert torch.all(confidence == 1.0)


def test_gru_variant_runs() -> None:
    cfg = ModelConfig(
        num_classes=4,
        embed_dim=48,
        tme=TmeConfig(in_features=14, seq_len=12, model_dim=32, variant="gru"),
        ife=IfeConfig(backbone="densenet121"),
        rbm=RbmConfig(cross_dim=32),
    )
    model = RegBridge(cfg)
    logits, _ = _run(model)
    assert logits.shape == (6, 4)


def test_ife_without_attention_runs() -> None:
    cfg = ModelConfig(
        num_classes=4,
        embed_dim=48,
        tme=TmeConfig(in_features=14, seq_len=12, model_dim=32, num_layers=2, num_heads=4),
        ife=IfeConfig(backbone="densenet121", use_attention=False),
        rbm=RbmConfig(cross_dim=32),
    )
    model = RegBridge(cfg)
    logits, _ = _run(model)
    assert logits.shape == (6, 4)
