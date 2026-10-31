"""Uncalibrated classification telemetry. This is never an action/intent object."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared.classes import vocabulary_for
from shared.desktop_protocol import Counter, WireModel


class ClassificationNotice(WireModel):
    state: Literal["warming_up", "rejected", "predicted", "error"]
    reason: Literal[
        "ready",
        "warming_up",
        "current_observation_invalid",
        "history_uninitialized",
        "stale",
        "insufficient_valid_ratio",
        "window_span_exceeded",
        "no_hand",
        "multiple_hands",
        "not_processed",
        "perception_error",
        "out_of_order",
        "future_input",
        "inference_error",
        "cancelled",
    ]
    model_version: str = Field(min_length=1, max_length=80)
    model_type: Literal["mlp", "tcn"]
    vocabulary_version: str = Field(min_length=1, max_length=40)
    synthetic_training: bool
    calibrated: Literal[False] = False
    stream_id: str = Field(min_length=1, max_length=128)
    context_epoch: Counter
    source_frame_id: Counter
    source_timestamp_ns: Counter
    track_id: str | None = Field(default=None, min_length=1, max_length=128)
    window_length: int = Field(ge=0, le=32)
    inference_ns: Counter | None = None
    logits: tuple[float, ...] = Field(max_length=15)

    @field_validator("calibrated", mode="before")
    @classmethod
    def uncalibrated(cls, value):
        if value is not False:
            raise ValueError("calibrated must be the boolean false")
        return value

    @model_validator(mode="after")
    def consistent(self):
        import math

        vocabulary = vocabulary_for(self.vocabulary_version)
        if (self.state == "predicted") != (self.reason == "ready"):
            raise ValueError("only a prediction may claim ready")
        if any(not math.isfinite(value) for value in self.logits):
            raise ValueError("nonfinite logits")
        if self.state == "predicted":
            if (
                self.reason != "ready"
                or len(self.logits) != vocabulary.class_count
                or self.track_id is None
                or self.window_length != 32
                or self.inference_ns is None
            ):
                raise ValueError("prediction requires complete same-frame provenance")
        elif self.logits or self.inference_ns is not None:
            raise ValueError("non-prediction cannot retain old scores")
        return self
