from __future__ import annotations

import torch

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoidal_motion_sequence(measures: torch.Tensor, seq_len: int) -> torch.Tensor:
    features = measures.shape[-1]
    time_axis = torch.linspace(0.0, 1.0, seq_len, device=measures.device).unsqueeze(-1)
    phases = torch.linspace(0.0, torch.pi / 2, features, device=measures.device).unsqueeze(0)
    basis = torch.sin(2.0 * torch.pi * time_axis + phases)
    return measures.unsqueeze(-2) * basis.unsqueeze(0)


def normalise_image(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[0]
    if channels != 3:
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (image - mean) / std
    mean = torch.tensor(_IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image - mean) / std


def radiograph_augment(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    out = image
    if torch.rand(1, generator=generator).item() < 0.5:
        out = torch.flip(out, dims=[-1])
    brightness = 1.0 + (torch.rand(1, generator=generator).item() - 0.5) * 0.4
    out = out * brightness
    if torch.rand(1, generator=generator).item() < 0.2:
        side = out.shape[-1]
        box = max(1, side // 8)
        top = int(torch.randint(0, side - box + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, side - box + 1, (1,), generator=generator).item())
        out = out.clone()
        out[:, top : top + box, left : left + box] = 0.0
    return out
