"""Explicit public-taxonomy application events; never legacy five-class gestures."""

from typing import Literal

from pydantic import Field, model_validator

from shared.contracts import Identifier, Index, Ns, SourceRecord
from shared.desktop_protocol import WireModel


class PublicGesturePolicy(WireModel):
    version: Literal["gesture-public-v1-draft"] = "gesture-public-v1-draft"
    status: Literal["draft"] = "draft"
    vocabulary_version: Literal["ipn-hands-v1"] = "ipn-hands-v1"
    class_on: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    neutral_hold_ns: Ns = 400_000_000
    cooldown_ns: Ns = 500_000_000
    max_age_ns: Ns = 150_000_000
    max_gap_ns: Ns = 100_000_000
    min_motion_ns: Ns = 150_000_000
    max_motion_ns: Ns = 1_500_000_000
    min_motion_samples: int = Field(default=3, strict=True, ge=2, le=128)
    min_displacement: float = Field(default=0.18, gt=0, le=1, allow_inf_nan=False)
    max_vertical_drift: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    max_reverse_step: float = Field(default=0.02, ge=0, le=0.1, allow_inf_nan=False)
    stationary_speed: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def bounded_timing(self):
        for name in (
            "neutral_hold_ns",
            "cooldown_ns",
            "max_age_ns",
            "max_gap_ns",
            "min_motion_ns",
            "max_motion_ns",
        ):
            if not 0 < getattr(self, name) <= 10_000_000_000:
                raise ValueError("public policy timing must be in (0,10s]")
        if self.min_motion_ns > self.max_motion_ns or self.max_gap_ns > self.max_age_ns:
            raise ValueError("invalid motion/freshness interval")
        return self


class PublicActionEvent(SourceRecord):
    event_id: Identifier
    track_id: Identifier
    context_epoch: Index
    source_token: Literal["TLT", "TRT"]
    action: Literal["previous_slide", "next_slide"]
    mode: Literal["presentation"] = "presentation"
    target_context: Identifier
    candidate_start_ns: Ns
    commit_ns: Ns
    model_version: Identifier
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["gesture-public-v1-draft"] = "gesture-public-v1-draft"
    synthetic: bool

    @model_validator(mode="after")
    def consistent(self):
        if not self.candidate_start_ns <= self.source_timestamp_ns <= self.commit_ns:
            raise ValueError("invalid public event timing")
        expected = "previous_slide" if self.source_token == "TLT" else "next_slide"
        if self.action != expected:
            raise ValueError("public token/action mapping mismatch")
        return self
