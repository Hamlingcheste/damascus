from __future__ import annotations

from typing import TypedDict

import torch


class ForgeBatch(TypedDict):
    motion: torch.Tensor
    image: torch.Tensor
    label: torch.Tensor


def move_batch(batch: ForgeBatch, device: torch.device) -> ForgeBatch:
    return {
        "motion": batch["motion"].to(device),
        "image": batch["image"].to(device),
        "label": batch["label"].to(device),
    }


def mask_modality(batch: ForgeBatch, drop: str) -> ForgeBatch:
    out: ForgeBatch = {
        "motion": batch["motion"].clone(),
        "image": batch["image"].clone(),
        "label": batch["label"],
    }
    if drop == "motion":
        out["motion"] = torch.zeros_like(out["motion"])
    elif drop == "image":
        out["image"] = torch.zeros_like(out["image"])
    return out
