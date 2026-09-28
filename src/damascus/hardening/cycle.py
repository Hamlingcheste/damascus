from __future__ import annotations

import torch

from damascus.smithy.settings import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.warm_restart_t0, T_mult=cfg.warm_restart_tmult
    )
