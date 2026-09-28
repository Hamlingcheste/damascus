from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from damascus.smithy.settings import DataConfig
from damascus.stock.cohorts import forge_split
from damascus.stock.quench_tank import normalise_image, radiograph_augment
from damascus.stock.work_order import ForgeBatch


class ForgeDataset(Dataset[ForgeBatch]):
    def __init__(self, split: dict[str, torch.Tensor], augment: bool, seed: int) -> None:
        self._motion = split["motion"]
        self._image = split["image"]
        self._label = split["label"]
        self._augment = augment
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return int(self._label.shape[0])

    def __getitem__(self, index: int) -> ForgeBatch:
        image = self._image[index]
        if self._augment:
            image = radiograph_augment(image, self._generator)
        image = normalise_image(image)
        return {
            "motion": self._motion[index],
            "image": image,
            "label": self._label[index],
        }


def _collate(items: list[ForgeBatch]) -> ForgeBatch:
    return {
        "motion": torch.stack([item["motion"] for item in items]),
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
    }


def build_datasets(cfg: DataConfig) -> tuple[ForgeDataset, ForgeDataset]:
    train_split = forge_split(cfg, cfg.num_train, cfg.seed)
    val_split = forge_split(cfg, cfg.num_val, cfg.seed + 1)
    train = ForgeDataset(train_split, augment=True, seed=cfg.seed)
    val = ForgeDataset(val_split, augment=False, seed=cfg.seed + 2)
    return train, val


def build_loaders(
    cfg: DataConfig, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader[ForgeBatch], DataLoader[ForgeBatch]]:
    train, val = build_datasets(cfg)
    train_loader: DataLoader[ForgeBatch] = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader: DataLoader[ForgeBatch] = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    return train_loader, val_loader
