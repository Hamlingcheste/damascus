from __future__ import annotations

import torch

from conftest import tiny_data_config, tiny_model_config
from damascus.blade import RegBridge
from damascus.smithy.hearth import set_seed
from damascus.stock.cohorts import forge_split


def test_cohort_generation_is_deterministic() -> None:
    first = forge_split(tiny_data_config(), 10, 123)
    second = forge_split(tiny_data_config(), 10, 123)
    assert torch.equal(first["motion"], second["motion"])
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["label"], second["label"])


def test_model_forward_is_deterministic_under_seed() -> None:
    split = forge_split(tiny_data_config(), 6, 0)
    set_seed(7)
    model_a = RegBridge(tiny_model_config()).eval()
    set_seed(7)
    model_b = RegBridge(tiny_model_config()).eval()
    with torch.no_grad():
        out_a = model_a(split["motion"], split["image"]).logits
        out_b = model_b(split["motion"], split["image"]).logits
    assert torch.allclose(out_a, out_b)
