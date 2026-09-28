from __future__ import annotations

from pathlib import Path

import msgspec
import torch

from conftest import tiny_full_config
from damascus.blade import RegBridge
from damascus.hardening.heat_treat import Smith
from damascus.smithy.hearth import set_seed
from damascus.stock.charge import build_loaders


def test_two_step_training_reduces_loss() -> None:
    set_seed(0)
    config = tiny_full_config()
    model = RegBridge(config.model)
    smith = Smith(model, config, torch.device("cpu"))
    train_loader, _ = build_loaders(config.data, config.train.batch_size)
    batch = next(iter(train_loader))
    first = smith.step(batch)["total"]
    for _ in range(30):
        last = smith.step(batch)["total"]
    assert last < first


def test_fit_writes_checkpoint(tmp_path: Path) -> None:
    set_seed(0)
    smith_config = msgspec.structs.replace(tiny_full_config(), output_dir=str(tmp_path / "run"))
    model = RegBridge(smith_config.model)
    smith = Smith(model, smith_config, torch.device("cpu"))
    train_loader, val_loader = build_loaders(smith_config.data, smith_config.train.batch_size)
    summary = smith.fit(train_loader, val_loader)
    assert "best_auroc" in summary
    assert (tmp_path / "run" / "last.pt").exists()
