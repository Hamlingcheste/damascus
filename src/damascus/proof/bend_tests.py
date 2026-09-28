from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from damascus.proof.edge_tests import classification_metrics, to_probabilities
from damascus.stock.work_order import ForgeBatch, mask_modality, move_batch


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader[ForgeBatch],
    device: torch.device,
    drop: str | None = None,
) -> dict[str, torch.Tensor]:
    model.eval()
    logits = []
    targets = []
    conf_motion = []
    conf_image = []
    for batch in loader:
        moved = move_batch(batch, device)
        if drop is not None:
            moved = mask_modality(moved, drop)
        out = model(moved["motion"], moved["image"])
        logits.append(out.logits.cpu())
        targets.append(moved["label"].cpu())
        conf_motion.append(out.confidence_motion.cpu())
        conf_image.append(out.confidence_image.cpu())
    return {
        "logits": torch.cat(logits),
        "targets": torch.cat(targets),
        "confidence_motion": torch.cat(conf_motion),
        "confidence_image": torch.cat(conf_image),
    }


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader[ForgeBatch], device: torch.device
) -> dict[str, float]:
    outputs = collect_outputs(model, loader, device)
    return classification_metrics(outputs["logits"], outputs["targets"])


def stagewise_auroc(
    model: torch.nn.Module,
    loader: DataLoader[ForgeBatch],
    device: torch.device,
    boundaries: list[tuple[int, int]],
) -> dict[str, float]:
    outputs = collect_outputs(model, loader, device)
    probs = to_probabilities(outputs["logits"])
    targets = outputs["targets"].numpy().astype(int)
    from sklearn.metrics import roc_auc_score

    result: dict[str, float] = {}
    for low, high in boundaries:
        mask = (targets <= low) | (targets >= high)
        if mask.sum() == 0:
            continue
        binary = (targets[mask] >= high).astype(int)
        positive_mass = probs[mask][:, high:].sum(axis=1)
        if np.unique(binary).size < 2:
            result[f"kl{low}_vs_kl{high}"] = float("nan")
            continue
        result[f"kl{low}_vs_kl{high}"] = float(roc_auc_score(binary, positive_mass))
    return result


def _add_motion_noise(batch: ForgeBatch, snr_db: float, generator: torch.Generator) -> ForgeBatch:
    motion = batch["motion"]
    signal_power = motion.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn(motion.shape, generator=generator) * noise_power.sqrt()
    return {"motion": motion + noise, "image": batch["image"], "label": batch["label"]}


@torch.no_grad()
def evaluate_robustness(
    model: torch.nn.Module,
    loader: DataLoader[ForgeBatch],
    device: torch.device,
    snr_levels: list[float],
    seed: int = 0,
) -> dict[str, float]:
    result: dict[str, float] = {"clean": evaluate_model(model, loader, device)["auroc"]}
    result["motion_zeroed"] = classification_metrics(
        collect_outputs(model, loader, device, drop="motion")["logits"],
        collect_outputs(model, loader, device, drop="motion")["targets"],
    )["auroc"]
    result["image_zeroed"] = classification_metrics(
        collect_outputs(model, loader, device, drop="image")["logits"],
        collect_outputs(model, loader, device, drop="image")["targets"],
    )["auroc"]
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    for snr in snr_levels:
        logits = []
        targets = []
        for batch in loader:
            noisy = _add_motion_noise(batch, snr, generator)
            moved = move_batch(noisy, device)
            out = model(moved["motion"], moved["image"])
            logits.append(out.logits.cpu())
            targets.append(moved["label"].cpu())
        result[f"motion_snr_{int(snr)}db"] = classification_metrics(
            torch.cat(logits), torch.cat(targets)
        )["auroc"]
    return result
