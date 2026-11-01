"""Opt-in HSP4: separately calibrated probabilities; raw v3 logits retain their meaning."""

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared.desktop_protocol import WireModel
from shared.model_preview_protocol import ModelPreviewCommand, ModelPreviewSnapshot
from shared.preview_protocol import (
    MAX_METADATA_BYTES,
    PREFIX,
    PreviewPacket,
    read_packet,
)


class ProbabilityNotice(WireModel):
    calibrated: Literal[True] = True
    status: Literal["draft"] = "draft"
    actions_enabled: Literal[False] = False
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: str = Field(min_length=1, max_length=80)
    synthetic: bool
    temperature: float = Field(ge=0.05, le=100, allow_inf_nan=False)
    threshold: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    probabilities: tuple[float, ...] = Field(min_length=1, max_length=15)
    rejection_status: Literal["accepted", "rejected", "unavailable"]

    @field_validator("calibrated", "actions_enabled", mode="before")
    @classmethod
    def exact_bools(cls, value, info):
        if value is not (info.field_name == "calibrated"):
            raise ValueError("calibration/action flags must be exact booleans")
        return value

    @model_validator(mode="after")
    def normalized(self):
        if any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in self.probabilities
        ) or not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("invalid calibrated probability distribution")
        expected = (
            "unavailable"
            if self.threshold is None
            else "accepted"
            if max(self.probabilities) >= self.threshold
            else "rejected"
        )
        if self.rejection_status != expected:
            raise ValueError("rejection status disagrees with calibrated threshold")
        return self


class CalibratedPreviewCommand(ModelPreviewCommand):
    version: Literal[4]


class CalibratedPreviewSnapshot(ModelPreviewSnapshot):
    version: Literal[4] = 4
    probability: ProbabilityNotice | None = None

    @model_validator(mode="after")
    def bound_probability(self):
        c, p = self.classification, self.probability
        predicted = c is not None and c.state == "predicted"
        if predicted != (p is not None):
            raise ValueError("probabilities only belong to a current prediction")
        if p is not None:
            if (
                p.model_version != c.model_version
                or len(p.probabilities) != len(c.logits)
                or c.synthetic_training
                and not p.synthetic
            ):
                raise ValueError(
                    "calibration model/vocabulary/synthetic binding mismatch"
                )
            top = max(c.logits)
            values = [math.exp((v - top) / p.temperature) for v in c.logits]
            total = sum(values)
            if any(
                not math.isclose(actual, value / total, abs_tol=1e-8)
                for actual, value in zip(p.probabilities, values)
            ):
                raise ValueError(
                    "probabilities do not match this frame's logits/temperature"
                )
        return self


class CalibratedPreviewPacket(PreviewPacket):
    def encode(self):
        metadata = self.snapshot.model_dump_json().encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds limit")
        return (
            PREFIX.pack(b"HSP4", len(metadata), len(self.jpeg)) + metadata + self.jpeg
        )


def read_calibrated_packet(stream):
    return read_packet(
        stream,
        expected_magic=b"HSP4",
        snapshot_type=CalibratedPreviewSnapshot,
        packet_type=CalibratedPreviewPacket,
    )
