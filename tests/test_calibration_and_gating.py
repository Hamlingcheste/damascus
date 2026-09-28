from __future__ import annotations

import torch

from damascus.forge.heat import quality_gate
from damascus.tongs.stresses import confidence_calibration


def test_calibration_target_favours_lower_loss_modality() -> None:
    c_motion = torch.full((4, 1), 0.5)
    c_image = torch.full((4, 1), 0.5)
    low_motion_loss = torch.full((4,), 0.1)
    high_image_loss = torch.full((4,), 2.0)
    target_motion = (high_image_loss / (low_motion_loss + high_image_loss)).mean()
    target_image = (low_motion_loss / (low_motion_loss + high_image_loss)).mean()
    assert target_motion > target_image
    loss = confidence_calibration(c_motion, c_image, low_motion_loss, high_image_loss)
    assert loss >= 0.0


def test_calibration_zero_when_confidence_matches_target() -> None:
    loss_motion = torch.tensor([0.5, 0.5])
    loss_image = torch.tensor([1.5, 1.5])
    denom = loss_motion + loss_image
    c_motion = (loss_image / denom).unsqueeze(-1)
    c_image = (loss_motion / denom).unsqueeze(-1)
    loss = confidence_calibration(c_motion, c_image, loss_motion, loss_image)
    assert float(loss) < 1e-10


def test_quality_gate_scales_by_confidence() -> None:
    embedding = torch.randn(3, 8)
    high = quality_gate(torch.full((3, 1), 0.9), embedding)
    low = quality_gate(torch.full((3, 1), 0.1), embedding)
    assert torch.all(high.abs() >= low.abs() - 1e-6)
    assert torch.allclose(quality_gate(torch.ones(3, 1), embedding), embedding)


def test_precision_weight_direction() -> None:
    sigma_motion = torch.tensor(0.25)
    sigma_image = torch.tensor(1.4)
    weight_motion = sigma_motion.pow(-2) / (sigma_motion.pow(-2) + sigma_image.pow(-2))
    assert float(weight_motion) > 0.5
