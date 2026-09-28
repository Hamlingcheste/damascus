from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from damascus.blade import RegBridge
from damascus.hardening.heat_treat import Smith
from damascus.proof.bend_tests import collect_outputs, evaluate_robustness, stagewise_auroc
from damascus.proof.edge_tests import classification_metrics, count_parameters
from damascus.smithy.hearth import get_logger, load_checkpoint, resolve_device
from damascus.smithy.settings import DamascusConfig, load_config
from damascus.stock.charge import build_loaders

app = typer.Typer(add_completion=False, help="RegBridge forge: confidence-gated multimodal fusion.")
_LOG = get_logger("damascus.anvil")

_CONFIG = typer.Option(..., "--config", "-c", help="path to a YAML experiment file")
_SET = typer.Option(None, "--set", "-s", help="override as section.key=value")


def _resolve(config_path: str, overrides: list[str] | None) -> DamascusConfig:
    return load_config(config_path, overrides or [])


@app.command()
def draw(config: str = _CONFIG, set_: list[str] | None = _SET) -> None:
    cfg = _resolve(config, set_)
    train_loader, _ = build_loaders(cfg.data, cfg.train.batch_size, cfg.train.num_workers)
    train_batch = next(iter(train_loader))
    summary = {
        "task": cfg.data.task,
        "train_items": cfg.data.num_train,
        "val_items": cfg.data.num_val,
        "motion_shape": list(train_batch["motion"].shape),
        "image_shape": list(train_batch["image"].shape),
        "num_classes": cfg.data.num_classes,
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def weld(config: str = _CONFIG, set_: list[str] | None = _SET) -> None:
    cfg = _resolve(config, set_)
    device = resolve_device(cfg.device)
    train_loader, val_loader = build_loaders(cfg.data, cfg.train.batch_size, cfg.train.num_workers)
    model = RegBridge(cfg.model)
    smith = Smith(model, cfg, device)
    summary = smith.fit(train_loader, val_loader)
    typer.echo(
        json.dumps(
            {
                "best_auroc": summary["best_auroc"],
                "best_epoch": summary["best_epoch"],
                "params_millions": count_parameters(model)["total_millions"],
                "output_dir": cfg.output_dir,
            },
            indent=2,
        )
    )


@app.command()
def proof(config: str = _CONFIG, set_: list[str] | None = _SET) -> None:
    cfg = _resolve(config, set_)
    device = resolve_device(cfg.device)
    _, val_loader = build_loaders(cfg.data, cfg.train.batch_size, cfg.train.num_workers)
    model = RegBridge(cfg.model).to(device)
    checkpoint = Path(cfg.output_dir) / "best.pt"
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, map_location=str(device))
        model.load_state_dict(state["model"])
    outputs = collect_outputs(model, val_loader, device)
    report = {
        "metrics": classification_metrics(outputs["logits"], outputs["targets"]),
        "stagewise": stagewise_auroc(model, val_loader, device, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "robustness": evaluate_robustness(model, val_loader, device, [20.0, 10.0, 5.0, 0.0]),
    }
    typer.echo(json.dumps(report, indent=2))


@app.command()
def edge(config: str = _CONFIG, set_: list[str] | None = _SET) -> None:
    cfg = _resolve(config, set_)
    device = resolve_device(cfg.device)
    _, val_loader = build_loaders(cfg.data, cfg.train.batch_size, cfg.train.num_workers)
    model = RegBridge(cfg.model).to(device)
    checkpoint = Path(cfg.output_dir) / "best.pt"
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, map_location=str(device))
        model.load_state_dict(state["model"])
    model.eval()
    batch = next(iter(val_loader))
    with torch.no_grad():
        out = model(batch["motion"].to(device), batch["image"].to(device))
    preds = out.logits.argmax(dim=-1).cpu().tolist()
    typer.echo(
        json.dumps(
            {
                "predictions": preds[:16],
                "confidence_motion": out.confidence_motion.squeeze(-1).cpu().tolist()[:16],
                "confidence_image": out.confidence_image.squeeze(-1).cpu().tolist()[:16],
            },
            indent=2,
        )
    )


@app.command()
def export(config: str = _CONFIG, set_: list[str] | None = _SET) -> None:
    cfg = _resolve(config, set_)
    device = resolve_device(cfg.device)
    model = RegBridge(cfg.model).to(device).eval()

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, inner: RegBridge) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, motion: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            return self.inner(motion, image).logits

    motion = torch.randn(1, cfg.model.tme.seq_len, cfg.model.tme.in_features, device=device)
    image = torch.randn(
        1, cfg.model.ife.in_channels, cfg.data.image_size, cfg.data.image_size, device=device
    )
    target = Path(cfg.output_dir) / "regbridge.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _LogitsOnly(model),
        (motion, image),
        str(target),
        input_names=["motion", "image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"motion": {0: "batch"}, "image": {0: "batch"}, "logits": {0: "batch"}},
    )
    typer.echo(json.dumps({"exported": str(target)}, indent=2))


if __name__ == "__main__":
    app()
