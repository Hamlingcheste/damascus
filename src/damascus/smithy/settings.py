from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import yaml


class TmeConfig(msgspec.Struct, frozen=True):
    in_features: int = 14
    seq_len: int = 32
    model_dim: int = 256
    num_layers: int = 6
    kernel_size: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    positional: bool = True
    variant: str = "tcn"


class IfeConfig(msgspec.Struct, frozen=True):
    in_channels: int = 3
    pretrained: bool = False
    attn_reduction: int = 16
    spatial_kernel: int = 7
    dropout: float = 0.2
    use_attention: bool = True
    backbone: str = "densenet201"


class RbmConfig(msgspec.Struct, frozen=True):
    cross_dim: int = 256
    use_confidence: bool = True
    use_gating: bool = True
    fusion: str = "bilinear"
    bilinear_directions: int = 2


class HeadConfig(msgspec.Struct, frozen=True):
    hidden_dim: int = 256
    dropout: float = 0.3


class ModelConfig(msgspec.Struct, frozen=True):
    num_classes: int = 5
    embed_dim: int = 512
    tme: TmeConfig = msgspec.field(default_factory=TmeConfig)
    ife: IfeConfig = msgspec.field(default_factory=IfeConfig)
    rbm: RbmConfig = msgspec.field(default_factory=RbmConfig)
    head: HeadConfig = msgspec.field(default_factory=HeadConfig)


class DataConfig(msgspec.Struct, frozen=True):
    name: str = "synthetic_oai"
    task: str = "kl5"
    num_train: int = 256
    num_val: int = 64
    num_classes: int = 5
    motion_features: int = 14
    motion_seq_len: int = 32
    image_size: int = 224
    image_channels: int = 3
    motion_noise: float = 1.0
    imaging_noise: float = 1.0
    domain_shift: float = 0.0
    seed: int = 42


class TrainConfig(msgspec.Struct, frozen=True):
    epochs: int = 120
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warm_restart_t0: int = 10
    warm_restart_tmult: int = 2
    early_stop_patience: int = 15
    label_smoothing: float = 0.1
    lambda_conf: float = 0.1
    aux_weight: float = 1.0
    amp: bool = True
    grad_clip: float | None = None
    ema_decay: float | None = None
    num_folds: int = 5
    num_workers: int = 0
    max_steps: int | None = None


class DamascusConfig(msgspec.Struct, frozen=True):
    name: str = "regbridge_oai_main"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs/regbridge"
    model: ModelConfig = msgspec.field(default_factory=ModelConfig)
    data: DataConfig = msgspec.field(default_factory=DataConfig)
    train: TrainConfig = msgspec.field(default_factory=TrainConfig)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_with_base(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"circular base reference at {resolved}")
    seen.add(resolved)
    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    refs = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict[str, Any] = {}
    for ref in refs:
        base_path = (resolved.parent / str(ref)).resolve()
        merged = _deep_merge(merged, _read_with_base(base_path, set(seen)))
    return _deep_merge(merged, raw)


def _apply_override(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def apply_overrides(tree: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = dict(tree)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, raw = item.split("=", 1)
        _apply_override(out, key.strip(), yaml.safe_load(raw))
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> DamascusConfig:
    tree = _read_with_base(Path(path), set())
    if overrides:
        tree = apply_overrides(tree, overrides)
    return msgspec.convert(tree, type=DamascusConfig, strict=False)


def config_to_dict(config: DamascusConfig) -> dict[str, Any]:
    return msgspec.to_builtins(config)
