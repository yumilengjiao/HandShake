"""T14 budgets are independent from the T02 feature/protocol configuration."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
import yaml

from shared.config import ConfigModel, UniqueKeyLoader

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, gt=0)]


class TrainConfig(ConfigModel):
    format_version: Literal["trainer-v1"] = "trainer-v1"
    model_type: Literal["mlp", "tcn"] = "tcn"
    model_variant: Literal["baseline", "motion", "dropout35", "motion_dropout35"] = "baseline"
    optimizer: Literal["AdamW"] = "AdamW"
    learning_rate: Positive = 0.001
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0001
    batch_size: Count = 64
    max_epochs: Count = 80
    patience: Count = 10
    min_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0001
    grad_clip_norm: Positive = 1.0
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] = 42
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda(:[0-9]+)?|mps)$")] = "cpu"
    threads: Count = 1
    sampling: Literal["uniform", "supervised_uniform"] = "uniform"
    class_weighting: Literal["none", "sqrt_inverse"] = "none"
    scheduler: Literal["constant", "plateau"] = "constant"
    lr_patience: Annotated[int, Field(strict=True, ge=0)] = 2
    lr_factor: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.5
    min_learning_rate: Positive = 0.000001
    checkpoint_metric: Literal["val_macro_f1"] = "val_macro_f1"
    shuffle_labels_diagnostic: bool = False

    @model_validator(mode="after")
    def scheduler_budget(self):
        if self.model_type != "tcn" and self.model_variant != "baseline":
            raise ValueError("experimental model variants are TCN-only")
        if self.scheduler == "plateau":
            if self.min_learning_rate > self.learning_rate:
                raise ValueError("min_learning_rate must not exceed learning_rate")
            if self.lr_patience + 1 >= self.patience:
                raise ValueError(
                    "patience must allow an epoch after the first LR reduction"
                )
        return self


def load_train_config(path=None):
    if path is None:
        return TrainConfig()
    return TrainConfig.model_validate(
        yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    )
