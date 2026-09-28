from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from damascus.blade import RegBridge
from damascus.hardening.cycle import build_optimizer, build_scheduler
from damascus.proof.bend_tests import evaluate_model
from damascus.smithy.hearth import get_logger, save_checkpoint, set_seed
from damascus.smithy.settings import DamascusConfig
from damascus.stock.work_order import ForgeBatch, move_batch
from damascus.tongs.stresses import total_loss


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            shadow_p.mul_(self.decay).add_(model_p, alpha=1.0 - self.decay)


class Smith:
    def __init__(self, model: RegBridge, config: DamascusConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_cfg = config.train
        self.optimizer = build_optimizer(self.model, self.train_cfg)
        self.scheduler = build_scheduler(self.optimizer, self.train_cfg)
        self.use_amp = self.train_cfg.amp and device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.use_amp)
        self.ema = (
            ModelEma(self.model, self.train_cfg.ema_decay) if self.train_cfg.ema_decay else None
        )
        self.logger = get_logger("damascus.smith")

    def step(self, batch: ForgeBatch) -> dict[str, float]:
        self.model.train()
        moved = move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=self.use_amp):
            out = self.model(moved["motion"], moved["image"])
            losses = total_loss(
                out.logits,
                out.aux_motion,
                out.aux_image,
                out.confidence_motion,
                out.confidence_image,
                moved["label"],
                label_smoothing=self.train_cfg.label_smoothing,
                lambda_conf=self.train_cfg.lambda_conf,
                aux_weight=self.train_cfg.aux_weight,
            )
        self.scaler.scale(losses["total"]).backward()
        if self.train_cfg.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update(self.model)
        return {key: float(value.detach()) for key, value in losses.items()}

    def fit(
        self,
        train_loader: DataLoader[ForgeBatch],
        val_loader: DataLoader[ForgeBatch],
    ) -> dict[str, object]:
        set_seed(self.config.seed)
        best_auroc = -1.0
        best_epoch = -1
        patience = 0
        history: list[dict[str, float]] = []
        global_step = 0
        max_steps = self.train_cfg.max_steps
        stop = False
        for epoch in range(self.train_cfg.epochs):
            running = 0.0
            seen = 0
            for batch in train_loader:
                losses = self.step(batch)
                running += losses["total"]
                seen += 1
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    stop = True
                    break
            self.scheduler.step()
            metrics = evaluate_model(self.model, val_loader, self.device)
            train_loss = running / max(1, seen)
            history.append(
                {"epoch": float(epoch), "train_loss": train_loss, "val_auroc": metrics["auroc"]}
            )
            self.logger.info(
                "epoch %d train_loss %.4f val_auroc %.4f", epoch, train_loss, metrics["auroc"]
            )
            current = metrics["auroc"]
            if current == current and current > best_auroc:
                best_auroc = current
                best_epoch = epoch
                patience = 0
                self._save("best.pt", epoch)
            else:
                patience += 1
                if patience >= self.train_cfg.early_stop_patience:
                    stop = True
            if stop:
                break
        self._save("last.pt", len(history) - 1)
        return {"best_auroc": best_auroc, "best_epoch": best_epoch, "history": history}

    def _save(self, name: str, epoch: int) -> None:
        path = Path(self.config.output_dir) / name
        save_checkpoint(path, self.model, self.optimizer, epoch, self.config.seed)


def cross_validate(config: DamascusConfig, device: torch.device) -> list[dict[str, object]]:
    import msgspec

    from damascus.smithy.settings import config_to_dict
    from damascus.stock.charge import build_loaders

    results: list[dict[str, object]] = []
    base_tree = config_to_dict(config)
    for fold in range(config.train.num_folds):
        set_seed(config.seed + fold)
        tree = copy.deepcopy(base_tree)
        tree["seed"] = config.seed + fold
        tree["data"]["seed"] = config.data.seed + fold
        tree["output_dir"] = f"{config.output_dir}/fold{fold}"
        fold_config = msgspec.convert(tree, type=DamascusConfig)
        train_loader, val_loader = build_loaders(
            fold_config.data, fold_config.train.batch_size, fold_config.train.num_workers
        )
        smith = Smith(RegBridge(fold_config.model), fold_config, device)
        summary = smith.fit(train_loader, val_loader)
        results.append({"fold": fold, **summary})
    return results
