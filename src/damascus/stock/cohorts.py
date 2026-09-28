from __future__ import annotations

import torch

from damascus.smithy.settings import DataConfig

_LATENT_DIM = 16
_SIGMA_LOW = 0.25
_SIGMA_HIGH = 1.4


def _stage_levels(num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        return torch.zeros(1)
    return torch.linspace(0.0, 1.0, num_classes)


def _class_means(num_classes: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(num_classes, _LATENT_DIM, generator=generator) * 2.0


def _projection(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(rows, cols, generator=generator) / (cols**0.5)


def _modality_sigmas(
    stage: float, motion_scale: float, imaging_scale: float
) -> tuple[float, float]:
    sigma_motion = (_SIGMA_LOW + (_SIGMA_HIGH - _SIGMA_LOW) * stage) * motion_scale
    sigma_image = (_SIGMA_LOW + (_SIGMA_HIGH - _SIGMA_LOW) * (1.0 - stage)) * imaging_scale
    return sigma_motion, sigma_image


def forge_split(cfg: DataConfig, count: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_classes = cfg.num_classes
    stages = _stage_levels(num_classes)
    means = _class_means(num_classes, generator)

    feat_proj = _projection(_LATENT_DIM, cfg.motion_features, generator)
    spatial_side = max(4, cfg.image_size // 16)
    code_dim = cfg.image_channels * spatial_side * spatial_side
    image_proj = _projection(_LATENT_DIM, code_dim, generator)

    labels = torch.randint(0, num_classes, (count,), generator=generator)
    motion = torch.empty(count, cfg.motion_seq_len, cfg.motion_features)
    image = torch.empty(count, cfg.image_channels, cfg.image_size, cfg.image_size)

    time_axis = torch.linspace(0.0, 1.0, cfg.motion_seq_len).unsqueeze(-1)
    phases = torch.linspace(0.0, torch.pi / 2, cfg.motion_features).unsqueeze(0)
    temporal_basis = torch.sin(2.0 * torch.pi * time_axis + phases)

    for i in range(count):
        label = int(labels[i].item())
        stage = float(stages[label].item())
        sigma_motion, sigma_image = _modality_sigmas(stage, cfg.motion_noise, cfg.imaging_noise)

        latent = means[label] + torch.randn(_LATENT_DIM, generator=generator)

        motion_feat = latent @ feat_proj
        motion_noise = torch.randn(cfg.motion_seq_len, cfg.motion_features, generator=generator)
        sequence = motion_feat.unsqueeze(0) * temporal_basis + sigma_motion * motion_noise
        motion[i] = sequence

        code = latent @ image_proj
        code = code + cfg.domain_shift
        grid = code.reshape(cfg.image_channels, spatial_side, spatial_side)
        upsampled = torch.nn.functional.interpolate(
            grid.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image_noise = torch.randn(
            cfg.image_channels, cfg.image_size, cfg.image_size, generator=generator
        )
        image[i] = upsampled + sigma_image * image_noise

    return {"motion": motion, "image": image, "label": labels}


def gait_signal_block(cfg: DataConfig, count: int, seed: int) -> dict[str, torch.Tensor]:
    return forge_split(cfg, count, seed)
