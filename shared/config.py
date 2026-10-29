"""Strict YAML configuration. Defaults are packaged once, with repo symlinks."""

from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
import yaml

from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION
from shared.contracts import ACTION_CAPABILITIES, GESTURE_IDS, Action, Capability, Gesture, Identifier, Index, Ns, Probability


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class RuntimeConfig(ConfigModel):
    mode: Literal["collect", "train", "evaluate", "replay", "dry_run", "live"] = "dry_run"
    enabled_on_start: Literal[False] = False
    backend: Literal["mock"] = "mock"
    max_hands: Annotated[int, Field(strict=True, ge=1, le=2)] = 2
    max_control_hands: Literal[1] = 1
    input_source: Literal["synthetic"] = "synthetic"
    target_context: Identifier = "mock-presentation"
    command_ttl_ns: Annotated[int, Field(strict=True, gt=0)] = 500_000_000


class InteractionConfig(ConfigModel):
    status: Literal["draft"] = "draft"
    wake_hold_ns: Annotated[int, Field(strict=True, gt=0)]
    exit_hold_ns: Annotated[int, Field(strict=True, gt=0)]
    neutral_hold_ns: Annotated[int, Field(strict=True, gt=0)]
    cooldown_ns: Ns
    max_prediction_age_ns: Annotated[int, Field(strict=True, gt=0)]


class GestureDefinition(ConfigModel):
    id: Index
    token: Gesture
    visible_definition: Identifier
    reference_rule: Identifier
    release_rule: Identifier


class GestureProtocol(ConfigModel):
    version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    status: Literal["draft"] = "draft"
    class_count: Literal[5] = 5
    coordinate_system: Literal["unmirrored_model_image"] = "unmirrored_model_image"
    palm_landmark_indices: list[Index]
    extended_pip_degrees_min: Annotated[int, Field(strict=True, gt=0, le=180)]
    folded_pip_degrees_max: Annotated[int, Field(strict=True, gt=0, le=180)]
    static_max_speed_widths_per_s: Probability
    static_max_gap_ns: Annotated[int, Field(strict=True, gt=0)]
    swipe_min_displacement_widths: Probability
    swipe_max_vertical_displacement_heights: Probability
    swipe_min_duration_ns: Annotated[int, Field(strict=True, gt=0)]
    swipe_max_duration_ns: Annotated[int, Field(strict=True, gt=0)]
    neutral_region_xy: Annotated[list[Probability], Field(min_length=4, max_length=4)]
    gestures: Annotated[list[GestureDefinition], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def check_protocol(self):
        if [(g.token, g.id) for g in self.gestures] != list(GESTURE_IDS.items()):
            raise ValueError("v1 gesture ids/order must be no_action=0 through exit_pose=4")
        if self.palm_landmark_indices != [0, 5, 9, 13, 17]:
            raise ValueError("v1 palm center uses landmarks 0,5,9,13,17")
        if self.folded_pip_degrees_max >= self.extended_pip_degrees_min:
            raise ValueError("folded and extended angle thresholds overlap")
        if self.swipe_min_duration_ns >= self.swipe_max_duration_ns:
            raise ValueError("swipe duration range is reversed")
        xmin, xmax, ymin, ymax = self.neutral_region_xy
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("neutral region must be [xmin,xmax,ymin,ymax]")
        return self


class ActionsConfig(ConfigModel):
    allowlist: Annotated[list[Action], Field(min_length=1)]
    capabilities: Annotated[list[Capability], Field(min_length=1)]
    mappings: dict[str, dict[Gesture, Action | None]]

    @model_validator(mode="after")
    def check_mappings(self):
        if not self.mappings or "presentation" not in self.mappings:
            raise ValueError("presentation mapping is required")
        if len(set(self.allowlist)) != len(self.allowlist) or len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate allowlist/capability")
        mapped = set()
        for mode, mapping in self.mappings.items():
            if not mode or set(mapping) != set(Gesture):
                raise ValueError("each mode must explicitly map all five gestures")
            for gesture, action in mapping.items():
                if gesture in {Gesture.NO_ACTION, Gesture.WAKE_OPEN, Gesture.EXIT_POSE} and action is not None:
                    raise ValueError("background/wake/exit are not device actions")
                if gesture in {Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT} and action is None:
                    raise ValueError("each swipe requires an explicit mapping")
                if action is not None:
                    if action not in self.allowlist or ACTION_CAPABILITIES[action] not in self.capabilities:
                        raise ValueError("mapped action must be allowed and capability-supported")
                    mapped.add(action)
        if mapped != set(self.allowlist):
            raise ValueError("allowlist must contain exactly the configured actions")
        return self


class TrainingConfig(ConfigModel):
    feature_schema_version: Literal[FEATURE_SCHEMA_VERSION] = FEATURE_SCHEMA_VERSION
    feature_dimension: Literal[94] = 94
    window_size: Literal[32] = 32
    class_count: Literal[5] = 5
    split_unit: Literal["subject"] = "subject"
    device: Literal["cpu"] = "cpu"
    seed: Index = 42
    fit_statistics_on: Literal["train_only"] = "train_only"


class PrivacyConfig(ConfigModel):
    consent_required: Literal[True] = True
    save_raw_video: StrictBool = False
    log_images: Literal[False] = False
    publish_identity: Literal[False] = False


class PlatformConfig(ConfigModel):
    platform_id: Identifier
    status: Literal["mock_only", "unverified"]
    capabilities: list[Capability]
    allowed_targets: Annotated[list[Identifier], Field(min_length=1)]
    independent_stop_verified: Literal[False] = False
    live_supported: Literal[False] = False


class ProjectConfig(ConfigModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    time_unit: Literal["ns"] = "ns"
    runtime: RuntimeConfig
    interaction: InteractionConfig
    gestures: GestureProtocol
    actions: ActionsConfig
    training: TrainingConfig
    privacy: PrivacyConfig
    platform: PlatformConfig

    @model_validator(mode="after")
    def check_platform(self):
        if self.runtime.mode == "live":
            raise ValueError("T02 has no live backend; live configuration is unavailable")
        if not set(self.actions.capabilities).issubset(self.platform.capabilities):
            raise ValueError("actions require unsupported platform capabilities")
        if self.runtime.target_context not in self.platform.allowed_targets:
            raise ValueError("target_context is not in platform allowlist")
        return self

    def digest(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(canonical.encode()).hexdigest()


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently discarding an earlier value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool)) or key in result:
            raise ValueError(f"invalid or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(directory: str | Path | None = None) -> ProjectConfig:
    """Load six named documents plus platform. No implicit fallback on bad files."""
    root = Path(directory) if directory is not None else files("shared").joinpath("defaults")
    documents = {}
    for name in ("runtime", "interaction", "gestures", "actions", "training", "privacy"):
        documents[name] = yaml.load(root.joinpath(f"{name}.yaml").read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    documents["platform"] = yaml.load(root.joinpath("platforms/mock.yaml").read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    return ProjectConfig.model_validate(documents)
