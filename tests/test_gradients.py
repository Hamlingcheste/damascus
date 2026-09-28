from __future__ import annotations

from conftest import tiny_data_config, tiny_model_config
from damascus.blade import RegBridge
from damascus.stock.cohorts import forge_split
from damascus.tongs.stresses import total_loss


def test_all_parameters_receive_gradient() -> None:
    model = RegBridge(tiny_model_config()).train()
    split = forge_split(tiny_data_config(), 8, 0)
    out = model(split["motion"], split["image"])
    losses = total_loss(
        out.logits,
        out.aux_motion,
        out.aux_image,
        out.confidence_motion,
        out.confidence_image,
        split["label"],
    )
    losses["total"].backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or float(param.grad.abs().sum()) == 0.0)
    ]
    assert missing == []
