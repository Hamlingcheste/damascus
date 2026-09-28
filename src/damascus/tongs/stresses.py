from __future__ import annotations

import torch
from torch.nn import functional as F


def smoothed_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 0.1
) -> torch.Tensor:
    num_classes = logits.shape[-1]
    log_prob = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        smooth = torch.full_like(log_prob, eps / num_classes)
        smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - eps + eps / num_classes)
    return -(smooth * log_prob).sum(dim=-1).mean()


def per_sample_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, reduction="none")


def confidence_calibration(
    c_motion: torch.Tensor,
    c_image: torch.Tensor,
    loss_motion: torch.Tensor,
    loss_image: torch.Tensor,
) -> torch.Tensor:
    denom = (loss_motion + loss_image).clamp_min(1e-8)
    target_motion = (loss_image / denom).detach()
    target_image = (loss_motion / denom).detach()
    motion_term = (c_motion.squeeze(-1) - target_motion) ** 2
    image_term = (c_image.squeeze(-1) - target_image) ** 2
    return (motion_term + image_term).mean()


def total_loss(
    class_logits: torch.Tensor,
    aux_motion_logits: torch.Tensor,
    aux_image_logits: torch.Tensor,
    c_motion: torch.Tensor,
    c_image: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.1,
    lambda_conf: float = 0.1,
    aux_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    classification = smoothed_cross_entropy(class_logits, target, label_smoothing)
    loss_motion = per_sample_ce(aux_motion_logits, target)
    loss_image = per_sample_ce(aux_image_logits, target)
    calibration = confidence_calibration(c_motion, c_image, loss_motion, loss_image)
    auxiliary = loss_motion.mean() + loss_image.mean()
    total = classification + lambda_conf * calibration + aux_weight * auxiliary
    return {
        "total": total,
        "classification": classification,
        "calibration": calibration,
        "auxiliary": auxiliary,
    }
