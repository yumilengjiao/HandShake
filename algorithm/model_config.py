"""Versioned draft model definitions; training budgets belong to T14."""

from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import yaml

from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION
from shared.classes import ClassVocabulary, vocabulary_for
from shared.config import ConfigModel, UniqueKeyLoader


class ModelConfig(ConfigModel):
    format_version: Literal["model-config-v1"] = "model-config-v1"
    status: Literal["draft"] = "draft"
    feature_schema_version: Literal[FEATURE_SCHEMA_VERSION] = FEATURE_SCHEMA_VERSION
    vocabulary_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    input_dim: Literal[94] = 94
    window_size: Literal[32] = 32
    dropout: Annotated[float, Field(strict=True, ge=0, lt=1, allow_inf_nan=False)] = 0.2


class MLPConfig(ModelConfig):
    model_type: Literal["mlp"] = "mlp"
    hidden_dims: tuple[int, int] = (128, 64)


class TCNConfig(ModelConfig):
    model_type: Literal["tcn"] = "tcn"
    channels: Literal[64] = 64
    kernel_size: Literal[3] = 3
    dilations: tuple[int, int, int, int] = (1, 2, 4, 8)


def load_model_config(model_type: str, path: str | Path | None = None) -> MLPConfig | TCNConfig:
    if model_type not in {"mlp", "tcn"}:
        raise ValueError("model_type must be mlp or tcn")
    resource = Path(path) if path else files("algorithm").joinpath(f"defaults/{model_type}.yaml")
    data = yaml.load(resource.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    config = (MLPConfig if model_type == "mlp" else TCNConfig).model_validate(data)
    if isinstance(config, MLPConfig) and config.hidden_dims != (128, 64):
        raise ValueError("the T12 baseline uses hidden dimensions 128,64")
    if isinstance(config, TCNConfig) and config.dilations != (1, 2, 4, 8):
        raise ValueError("the T13 baseline uses dilations 1,2,4,8")
    return config


MODEL_VARIANTS = ("baseline", "motion", "dropout35", "motion_dropout35")


def variant_config(model_type, path=None, *, variant="baseline"):
    """Experimental variants are explicit; packaged baseline defaults stay intact."""
    config = load_model_config(model_type, path)
    if variant not in MODEL_VARIANTS or (variant != "baseline" and model_type != "tcn"):
        raise ValueError("experimental model variants are TCN-only")
    if variant in {"dropout35", "motion_dropout35"}:
        config = config.model_copy(update={"dropout": 0.35})
    return config


def model_architecture(model_type, *, vocabulary=None, variant="baseline"):
    config = variant_config(model_type, variant=variant)
    vocabulary = vocabulary or vocabulary_for()
    architecture = config.model_dump(mode="json") | {
        "vocabulary_version": vocabulary.vocabulary_version,
        "num_classes": vocabulary.class_count,
    }
    if variant != "baseline":
        architecture["model_variant"] = variant
    if variant in {"motion", "motion_dropout35"}:
        architecture["motion_branch"] = {
            "format_version": "causal-motion-contrast-v1",
            "scales": [4, 8, 16],
            "geometry_channels": 69,
            "hidden_channels": 64,
            "presence_mask": "any_positive_keypoint_mask_73_to_93",
            "fusion": "zero_initialized_additive_class_logits",
        }
    return architecture


def build_model(model_type: str, path: str | Path | None = None, *, vocabulary: ClassVocabulary | None = None,
                variant="baseline"):
    config = variant_config(model_type, path, variant=variant)
    vocabulary = vocabulary or vocabulary_for()
    if isinstance(config, MLPConfig):
        from algorithm.models.mlp import GestureMLP
        return GestureMLP(input_dim=config.input_dim, hidden_dims=config.hidden_dims, dropout=config.dropout, vocabulary=vocabulary)
    if variant in {"motion", "motion_dropout35"}:
        from algorithm.models.motion_tcn import MotionGestureTCN
        model_class = MotionGestureTCN
    else:
        from algorithm.models.tcn import GestureTCN
        model_class = GestureTCN
    return model_class(input_dim=config.input_dim, channels=config.channels, kernel_size=config.kernel_size,
                       dilations=config.dilations, dropout=config.dropout, window_size=config.window_size, vocabulary=vocabulary)
