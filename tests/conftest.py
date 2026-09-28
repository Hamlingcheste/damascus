from __future__ import annotations

import pytest

from damascus.smithy.settings import (
    DamascusConfig,
    DataConfig,
    HeadConfig,
    IfeConfig,
    ModelConfig,
    RbmConfig,
    TmeConfig,
    TrainConfig,
)


def tiny_model_config(num_classes: int = 4, fusion: str = "bilinear") -> ModelConfig:
    return ModelConfig(
        num_classes=num_classes,
        embed_dim=48,
        tme=TmeConfig(
            in_features=14, seq_len=12, model_dim=32, num_layers=2, num_heads=4, dropout=0.0
        ),
        ife=IfeConfig(in_channels=3, backbone="densenet121", dropout=0.0),
        rbm=RbmConfig(cross_dim=32, fusion=fusion),
        head=HeadConfig(hidden_dim=32, dropout=0.0),
    )


def tiny_data_config(num_classes: int = 4) -> DataConfig:
    return DataConfig(
        num_classes=num_classes,
        motion_features=14,
        motion_seq_len=12,
        image_size=64,
        image_channels=3,
        num_train=16,
        num_val=12,
        seed=0,
    )


def tiny_full_config() -> DamascusConfig:
    return DamascusConfig(
        name="pytest",
        seed=0,
        device="cpu",
        output_dir="runs/pytest",
        model=tiny_model_config(),
        data=tiny_data_config(),
        train=TrainConfig(
            epochs=1,
            batch_size=4,
            amp=False,
            num_folds=1,
            max_steps=2,
            early_stop_patience=5,
        ),
    )


@pytest.fixture()
def model_config() -> ModelConfig:
    return tiny_model_config()


@pytest.fixture()
def data_config() -> DataConfig:
    return tiny_data_config()


@pytest.fixture()
def full_config() -> DamascusConfig:
    return tiny_full_config()
