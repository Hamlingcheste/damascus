from __future__ import annotations

import torch

from conftest import tiny_data_config, tiny_model_config
from damascus.blade import RegBridge
from damascus.stock.cohorts import forge_split
from damascus.stock.work_order import mask_modality


def test_zeroed_modality_keeps_valid_output() -> None:
    model = RegBridge(tiny_model_config()).eval()
    split = forge_split(tiny_data_config(), 8, 3)
    batch = {"motion": split["motion"], "image": split["image"], "label": split["label"]}
    for drop in ("motion", "image"):
        masked = mask_modality(batch, drop)
        with torch.no_grad():
            out = model(masked["motion"], masked["image"])
        assert torch.isfinite(out.logits).all()
        probs = torch.softmax(out.logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-5)


def test_missing_modality_changes_prediction_distribution() -> None:
    model = RegBridge(tiny_model_config()).eval()
    split = forge_split(tiny_data_config(), 8, 5)
    with torch.no_grad():
        full = model(split["motion"], split["image"]).logits
        zeroed = model(torch.zeros_like(split["motion"]), split["image"]).logits
    assert not torch.allclose(full, zeroed)
