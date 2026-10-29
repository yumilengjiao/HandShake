"""T02 v1 contracts. All times are explicit integer nanoseconds in one timebase.

Lists make the initial format portable and JSON serializable. Capture/feature adapters
must convert their arrays at this boundary; performance optimization belongs to T03/T11.
"""

from enum import Enum
from math import isclose
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION

Ns = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Index = Annotated[int, Field(strict=True, ge=0)]
Identifier = Annotated[str, Field(strict=True, min_length=1)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Timebase = Literal["monotonic", "recording_relative", "synthetic"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    time_unit: Literal["ns"] = "ns"
    timebase: Timebase = "monotonic"


class Gesture(str, Enum):
    NO_ACTION = "no_action"
    WAKE_OPEN = "wake_open"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    EXIT_POSE = "exit_pose"


GESTURE_IDS = {gesture: index for index, gesture in enumerate(Gesture)}


class Capability(str, Enum):
    PRESENTATION = "presentation"
    MEDIA = "media"
    VOLUME = "volume"
    POINTER = "pointer"
    SCROLL = "scroll"


class Action(str, Enum):
    PREVIOUS_SLIDE = "previous_slide"
    NEXT_SLIDE = "next_slide"
    PREVIOUS_TRACK = "previous_track"
    NEXT_TRACK = "next_track"
    PLAY_PAUSE = "play_pause"
    SET_VOLUME = "set_volume"
    MOVE_POINTER = "move_pointer"
    CLICK = "click"
    SCROLL = "scroll"
    BUTTON_DOWN = "button_down"
    BUTTON_UP = "button_up"


ACTION_CAPABILITIES = {
    Action.PREVIOUS_SLIDE: Capability.PRESENTATION,
    Action.NEXT_SLIDE: Capability.PRESENTATION,
    Action.PREVIOUS_TRACK: Capability.MEDIA,
    Action.NEXT_TRACK: Capability.MEDIA,
    Action.PLAY_PAUSE: Capability.MEDIA,
    Action.SET_VOLUME: Capability.VOLUME,
    Action.MOVE_POINTER: Capability.POINTER,
    Action.CLICK: Capability.POINTER,
    Action.SCROLL: Capability.SCROLL,
    Action.BUTTON_DOWN: Capability.POINTER,
    Action.BUTTON_UP: Capability.POINTER,
}


class TransformMetadata(Contract):
    """Model coordinates are never preview-mirrored; x right, y down, z relative."""

    model_mirrored: Literal[False] = False
    preview_mirrored: StrictBool = True
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    coordinate_system: Literal["normalized_image_xy_relative_z"] = "normalized_image_xy_relative_z"


class SourceRecord(Contract):
    stream_id: Identifier
    source_frame_id: Index
    source_timestamp_ns: Ns


class FramePacket(Contract):
    stream_id: Identifier
    frame_id: Index
    capture_timestamp_ns: Ns
    # Canonical RGB uint8, [height][width][3]; no implicit channel order.
    image: list[list[list[Annotated[int, Field(strict=True, ge=0, le=255)]]]]
    image_size: Annotated[list[Annotated[int, Field(strict=True, gt=0)]], Field(min_length=2, max_length=2)]
    pixel_format: Literal["RGB8"] = "RGB8"
    transform_metadata: TransformMetadata

    @model_validator(mode="after")
    def check_image(self):
        width, height = self.image_size
        if len(self.image) != height or any(len(row) != width for row in self.image):
            raise ValueError("image shape must match image_size=[width,height]")
        if any(len(pixel) != 3 for row in self.image for pixel in row):
            raise ValueError("RGB8 requires exactly 3 channels")
        return self


class PerceptionStatus(str, Enum):
    HANDS = "processed_hands"
    NO_HAND = "processed_no_hand"
    NOT_PROCESSED = "not_processed"
    ERROR = "error"


class FrameStatus(SourceRecord):
    perception_status: PerceptionStatus
    processed_at_ns: Ns | None = None
    drop_or_error_reason: Identifier | None = None

    @model_validator(mode="after")
    def check_processing(self):
        if self.perception_status != PerceptionStatus.NOT_PROCESSED and self.processed_at_ns is None:
            raise ValueError("processed frames require processed_at_ns")
        if self.processed_at_ns is not None and self.processed_at_ns < self.source_timestamp_ns:
            raise ValueError("processed_at_ns precedes the source timestamp")
        if self.perception_status in {PerceptionStatus.NOT_PROCESSED, PerceptionStatus.ERROR}:
            if not self.drop_or_error_reason:
                raise ValueError("unprocessed/error frames require a reason")
        elif self.drop_or_error_reason is not None:
            raise ValueError("successful perception cannot carry a drop/error reason")
        return self


class HandObservation(SourceRecord):
    track_id: Identifier
    landmarks_image: Annotated[list[Annotated[list[Finite], Field(min_length=3, max_length=3)]], Field(min_length=21, max_length=21)]
    validity_mask: Annotated[list[StrictBool], Field(min_length=21, max_length=21)]
    observation_valid: StrictBool
    handedness: Literal["left", "right", "unknown"]
    handedness_score: Probability | None = None
    quality: Probability
    quality_flags: list[Identifier] = Field(default_factory=list)
    transform_metadata: TransformMetadata

    @model_validator(mode="after")
    def check_validity(self):
        if self.observation_valid and not all(self.validity_mask):
            raise ValueError("v1 valid observations require all 21 valid landmarks")
        if not self.observation_valid and not self.quality_flags:
            raise ValueError("invalid observations require quality_flags")
        return self


class FeatureWindow(Contract):
    stream_id: Identifier
    track_id: Identifier
    context_epoch: Index
    source_frame_id: Index
    timestamps_ns: Annotated[list[Ns], Field(min_length=1)]
    features: Annotated[list[Annotated[list[Finite], Field(min_length=94, max_length=94)]], Field(min_length=1)]
    feature_schema_version: Literal[FEATURE_SCHEMA_VERSION] = FEATURE_SCHEMA_VERSION
    ready: StrictBool
    ready_reason: Identifier

    @model_validator(mode="after")
    def check_window(self):
        if len(self.timestamps_ns) != len(self.features):
            raise ValueError("timestamps and feature rows must have the same T")
        if any(a >= b for a, b in zip(self.timestamps_ns, self.timestamps_ns[1:])):
            raise ValueError("window timestamps must strictly increase")
        return self


class ModelPrediction(SourceRecord):
    track_id: Identifier
    context_epoch: Index
    inference_timestamp_ns: Ns
    logits: Annotated[list[Finite], Field(min_length=5, max_length=5)]
    calibrated_probabilities: Annotated[list[Probability], Field(min_length=5, max_length=5)] | None = None
    model_version: Identifier
    vocabulary_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    intent_score: Probability | None = None
    phase_scores: dict[Literal["background", "preparation", "active", "release"], Probability] | None = None

    @model_validator(mode="after")
    def check_prediction(self):
        if self.inference_timestamp_ns < self.source_timestamp_ns:
            raise ValueError("inference precedes source time")
        if self.calibrated_probabilities is not None and not isclose(sum(self.calibrated_probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("calibrated probabilities must sum to 1")
        if self.phase_scores is not None and (len(self.phase_scores) != 4 or not isclose(sum(self.phase_scores.values()), 1.0, abs_tol=1e-6)):
            raise ValueError("phase scores require four normalized probabilities")
        return self


class GestureEvent(SourceRecord):
    event_id: Identifier
    track_id: Identifier
    context_epoch: Index
    gesture_id: Gesture
    candidate_start_timestamp_ns: Ns
    commit_timestamp_ns: Ns
    mode: Identifier
    decision_reason: Identifier

    @model_validator(mode="after")
    def check_event(self):
        if self.gesture_id == Gesture.NO_ACTION:
            raise ValueError("no_action cannot emit an event")
        if not self.candidate_start_timestamp_ns <= self.source_timestamp_ns <= self.commit_timestamp_ns:
            raise ValueError("event times must be candidate <= source <= commit")
        return self


class ActionCommand(Contract):
    command_id: Identifier
    source_event_id: Identifier
    stream_id: Identifier
    context_epoch: Index
    action: Action
    arguments: dict[str, Any] = Field(default_factory=dict)
    issued_at_ns: Ns
    expires_at_ns: Ns
    required_capabilities: Annotated[list[Capability], Field(min_length=1)]
    target_context: Identifier
    control_stream_id: Identifier | None = None
    update_seq: Index | None = None

    @model_validator(mode="after")
    def check_command(self):
        if self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("command expiry must follow issue time")
        if ACTION_CAPABILITIES[self.action] not in self.required_capabilities:
            raise ValueError("missing capability required by action")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise ValueError("duplicate capabilities")
        if (self.control_stream_id is None) != (self.update_seq is None):
            raise ValueError("control_stream_id and update_seq must appear together")
        if self.action in {Action.MOVE_POINTER, Action.SET_VOLUME, Action.SCROLL} and self.control_stream_id is None:
            raise ValueError("continuous updates require control_stream_id and update_seq")
        if self.action == Action.MOVE_POINTER:
            _check_arguments(self.arguments, {"x", "y"})
            if any(not 0 <= value <= 1 for value in self.arguments.values()):
                raise ValueError("pointer coordinates must be normalized in [0,1]")
        elif self.action == Action.SET_VOLUME:
            _check_arguments(self.arguments, {"value"})
            if not 0 <= self.arguments["value"] <= 1:
                raise ValueError("absolute volume must be in [0,1]")
        elif self.action == Action.SCROLL:
            _check_arguments(self.arguments, {"delta"})
        elif self.arguments:
            raise ValueError("v1 discrete actions take no arguments")
        return self


def _check_arguments(arguments: dict, keys: set[str]) -> None:
    from math import isfinite

    if set(arguments) != keys or any(type(v) not in {int, float} or not isfinite(v) for v in arguments.values()):
        raise ValueError(f"expected finite numeric arguments: {sorted(keys)}")


class InjectionStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class ActionResult(Contract):
    command_id: Identifier
    accepted: StrictBool
    injection_status: InjectionStatus
    error_code: Identifier | None = None
    completed_at_ns: Ns
    acknowledgement_level: Literal["none", "mock", "injection", "application"]

    @model_validator(mode="after")
    def check_result(self):
        if not self.accepted and self.injection_status != InjectionStatus.NOT_ATTEMPTED:
            raise ValueError("rejected commands cannot have attempted injection")
        if not self.accepted and (self.error_code is None or self.acknowledgement_level != "none"):
            raise ValueError("rejected commands require a reason and no acknowledgement")
        if self.injection_status in {InjectionStatus.FAILED, InjectionStatus.UNCERTAIN} and self.error_code is None:
            raise ValueError("failed/uncertain injection requires error_code")
        if self.injection_status == InjectionStatus.CONFIRMED:
            if self.acknowledgement_level not in {"injection", "application"} or self.error_code is not None:
                raise ValueError("confirmed injection requires injection/application acknowledgement")
        elif self.acknowledgement_level in {"injection", "application"}:
            raise ValueError("injection/application acknowledgement requires confirmed injection")
        return self


class RuntimeContext(Contract):
    stream_id: Identifier
    context_epoch: Index
    enabled: StrictBool = False
    mode: Identifier = "presentation"
    target_context: Identifier = "mock-presentation"
    owner_track_id: Identifier | None = None
    suspended: StrictBool = False


class ContextChange(Contract):
    stream_id: Identifier
    context_epoch: Index
    timestamp_ns: Ns
    reason: Literal["start", "enable", "disable", "mode", "model", "owner", "suspend", "stop", "reconnect"]


class HealthEvent(Contract):
    stream_id: Identifier
    context_epoch: Index
    timestamp_ns: Ns
    component: Identifier
    status: Literal["ok", "degraded", "failed"]
    reason: Identifier
