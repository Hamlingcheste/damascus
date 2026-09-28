from __future__ import annotations

from pathlib import Path

from damascus.smithy.settings import apply_overrides, load_config

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiment"


def test_main_matches_reported_hyperparameters() -> None:
    cfg = load_config(_CONFIGS / "main.yaml")
    assert cfg.train.batch_size == 64
    assert cfg.train.epochs == 120
    assert cfg.train.weight_decay == 1.0e-2
    assert cfg.train.warm_restart_t0 == 10 and cfg.train.warm_restart_tmult == 2
    assert cfg.train.label_smoothing == 0.1 and cfg.train.lambda_conf == 0.1
    assert cfg.model.embed_dim == 512
    assert cfg.model.tme.num_layers == 6 and cfg.model.tme.model_dim == 256
    assert cfg.model.tme.num_heads == 8
    assert cfg.model.rbm.cross_dim == 256
    assert cfg.model.ife.attn_reduction == 16


def test_base_merge_and_ablation_override() -> None:
    base = load_config(_CONFIGS / "main.yaml")
    concat = load_config(_CONFIGS / "ablation_a_concat.yaml")
    assert base.model.rbm.fusion == "bilinear"
    assert concat.model.rbm.fusion == "concat"
    assert concat.model.embed_dim == base.model.embed_dim


def test_cli_style_override() -> None:
    cfg = load_config(_CONFIGS / "_smoke.yaml", ["train.lr=0.002", "model.rbm.fusion=cross_attn"])
    assert cfg.train.lr == 0.002
    assert cfg.model.rbm.fusion == "cross_attn"


def test_apply_overrides_creates_nested_keys() -> None:
    tree = apply_overrides({"train": {"lr": 1.0}}, ["train.lr=2.0", "data.seed=9"])
    assert tree["train"]["lr"] == 2.0
    assert tree["data"]["seed"] == 9


def test_smoke_config_loads() -> None:
    cfg = load_config(_CONFIGS / "_smoke.yaml")
    assert cfg.train.max_steps == 2
    assert cfg.device == "cpu"
