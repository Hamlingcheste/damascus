from __future__ import annotations

import torch

from conftest import tiny_data_config, tiny_model_config
from damascus.blade import RegBridge
from damascus.smithy.hearth import set_seed
from damascus.stock.cohorts import forge_split
from damascus.tongs.stresses import smoothed_cross_entropy


def test_single_batch_overfit() -> None:
    set_seed(0)
    model = RegBridge(tiny_model_config()).train()
    split = forge_split(tiny_data_config(), 8, 0)
    motion, image, label = split["motion"], split["image"], split["label"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for stepIndex in range(60):
        optimizer.zero_grad()
        out = model(motion, image)
        loss = smoothed_cross_entropy(out.logits, label, eps=0.0)
        loss.backward()
        optimizer.step()
        if stepIndex == 0:
            first = float(loss.detach())
    final = float(loss.detach())
    assert first is not None and final < first * 0.5
    model.eval()
    with torch.no_grad():
        preds = model(motion, image).logits.argmax(dim=-1)
    assert (preds == label).float().mean() >= 0.75
