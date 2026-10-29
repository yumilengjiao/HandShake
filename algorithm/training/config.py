"""T14 budgets are independent from the T02 feature/protocol configuration."""
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import yaml

from shared.config import ConfigModel, UniqueKeyLoader

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, gt=0)]


class TrainConfig(ConfigModel):
    format_version: Literal["trainer-v1"] = "trainer-v1"
    model_type: Literal["mlp", "tcn"] = "tcn"
    optimizer: Literal["AdamW"] = "AdamW"
    learning_rate: Positive = 0.001
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0001
    batch_size: Count = 64
    max_epochs: Count = 80
    patience: Count = 10
    min_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0001
    grad_clip_norm: Positive = 1.0
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32-1)] = 42
    device: Literal["cpu"] = "cpu"
    threads: Count = 1
    sampling: Literal["uniform"] = "uniform"
    scheduler: Literal["constant"] = "constant"
    checkpoint_metric: Literal["val_macro_f1"] = "val_macro_f1"
    shuffle_labels_diagnostic: bool = False


def load_train_config(path=None):
    if path is None:
        return TrainConfig()
    return TrainConfig.model_validate(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader))
