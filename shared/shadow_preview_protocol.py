"""HSP5 native shadow session. Lifecycle epoch and model/owner epoch are DISTINCT."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared.calibrated_preview_protocol import ProbabilityNotice
from shared.desktop_protocol import Counter, SessionId, WireModel
from shared.inference_protocol import ClassificationNotice
from shared.preview_protocol import (
    MAX_METADATA_BYTES,
    PREFIX,
    PreviewPacket,
    PreviewSnapshot,
    read_packet,
)


class ShadowCommand(WireModel):
    version: Literal[5]
    session_id: SessionId
    command_id: Counter
    epoch: Counter
    kind: Literal[
        "start_camera",
        "stop",
        "shutdown",
        "heartbeat",
        "surface_ack",
        "enable_shadow",
        "request_owner",
    ]
    model_epoch: Counter | None = None
    model_sequence: Counter | None = None
    model_revision: Counter | None = None
    track_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def controls(self):
        model_fields = (self.model_epoch, self.model_sequence, self.model_revision)
        if any(v is not None for v in model_fields) != all(
            v is not None for v in model_fields
        ):
            raise ValueError("model context must be complete")
        required = self.kind in {"surface_ack", "enable_shadow", "request_owner"}
        if required and self.model_epoch is None:
            raise ValueError("shadow control requires captured model context")
        if (
            self.kind in {"start_camera", "stop", "shutdown"}
            and self.model_epoch is not None
        ):
            raise ValueError("source lifecycle does not take model context")
        if (self.kind == "request_owner") != (self.track_id is not None):
            raise ValueError("only owner requests name a track")
        return self


class ShadowCandidate(WireModel):
    event_id: str = Field(min_length=1, max_length=256)
    action: Literal["next_slide", "previous_slide"]
    source_frame_id: Counter
    source_timestamp_ns: Counter
    model_epoch: Counter


class ShadowInfo(WireModel):
    execution: Literal["shadow"] = "shadow"
    actions_enabled: Literal[False] = False
    actual_page: Literal[1] = 1
    model_epoch: Counter
    model_sequence: Counter
    model_revision: Counter
    model_stream_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=80)
    state: Literal[
        "DISABLED", "WAIT_OWNER", "WAIT_RELEASE", "ACTIVE", "CANDIDATE", "SUSPENDED"
    ]
    reason: str = Field(min_length=1, max_length=256)
    model_warming: bool
    model_busy: bool
    surface_ready: bool
    eligible_owner: str | None = Field(default=None, min_length=1, max_length=128)
    owner_track_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_request_pending: bool
    candidates_total: Counter
    candidate: ShadowCandidate | None = None
    synthetic: bool
    control_status: Literal["none", "accepted", "rejected"] = "none"

    @field_validator("actions_enabled", mode="before")
    @classmethod
    def no_authority(cls, value):
        if value is not False:
            raise ValueError("shadow cannot authorize actions")
        return value

    @field_validator("actual_page", mode="before")
    @classmethod
    def fixed_page(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("shadow page must remain one")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if int(self.model_revision) != 0:
            raise ValueError("shadow has no application revisions")
        if self.candidate is not None and (
            self.candidate.model_epoch != self.model_epoch
            or int(self.candidates_total) == 0
        ):
            raise ValueError("candidate belongs to another model epoch")
        if self.state in {"DISABLED", "SUSPENDED"} and (
            self.owner_track_id is not None
            or self.eligible_owner is not None
            or self.owner_request_pending
            or self.model_warming
            or self.candidate is not None
        ):
            raise ValueError("inactive shadow cannot retain owner/offer/candidate")
        if self.eligible_owner is not None and (
            self.state != "WAIT_OWNER"
            or self.owner_track_id is not None
            or self.owner_request_pending
        ):
            raise ValueError("owner offer requires an unowned active preview")
        if (
            self.state in {"WAIT_RELEASE", "ACTIVE", "CANDIDATE"}
            and self.owner_track_id is None
            or self.state == "WAIT_OWNER"
            and self.owner_track_id is not None
            or self.owner_request_pending
            and self.state != "WAIT_OWNER"
        ):
            raise ValueError("shadow phase/owner mismatch")
        if self.model_warming and (
            self.owner_track_id is None or self.state != "WAIT_RELEASE"
        ):
            raise ValueError("model handover requires a bound owner")
        return self


class ShadowSnapshot(PreviewSnapshot):
    version: Literal[5] = 5
    model_status: Literal["unavailable", "loaded_shadow", "error"] = "unavailable"
    classification: ClassificationNotice | None = None
    probability: ProbabilityNotice | None = None
    shadow: ShadowInfo | None = None

    @field_validator("control_available", mode="before")
    @classmethod
    def no_control(cls, value):
        if value is not False:
            raise ValueError("shadow preview has no execution capability")
        return value

    @model_validator(mode="after")
    def same_frame_and_context(self):
        c, p, s = self.classification, self.probability, self.shadow
        if s is not None and (
            self.state != "preview"
            or self.source != "camera"
            or self.model_status != "loaded_shadow"
        ):
            raise ValueError("shadow state requires a live observed preview")
        if (self.model_status == "loaded_shadow") != (s is not None):
            raise ValueError("loaded shadow requires explicit session state")
        if c is not None:
            if s is not None and s.state in {"DISABLED", "SUSPENDED"}:
                raise ValueError("inactive model must clear its classification")
            if (
                s is None
                or c.context_epoch != s.model_epoch
                or c.model_version != s.model_version
                or c.stream_id != s.model_stream_id
                or c.source_frame_id != self.frame_id
                or c.source_timestamp_ns != self.frame_timestamp_ns
            ):
                raise ValueError(
                    "classification must match frame AND model epoch, not lifecycle epoch"
                )
            if c.vocabulary_version != "ipn-hands-v1":
                raise ValueError("native shadow session requires the public vocabulary")
            if c.state == "predicted" and (
                len(self.hands) != 1
                or not self.hands[0].usable
                or not all(self.hands[0].validity)
            ):
                raise ValueError("predictions require one valid observed hand")
        if (c is not None and c.state == "predicted") != (p is not None):
            raise ValueError("probability belongs only to a current prediction")
        if p is not None:
            if s.model_warming:
                raise ValueError("model warmup cannot retain a prediction")
            import math

            if (
                p.model_version != c.model_version
                or len(p.probabilities) != len(c.logits)
                or c.synthetic_training
                and not p.synthetic
                or p.synthetic
                and not s.synthetic
            ):
                raise ValueError("calibration/model provenance mismatch")
            scaled = [math.exp((v - max(c.logits)) / p.temperature) for v in c.logits]
            if any(
                not math.isclose(a, b / sum(scaled), abs_tol=1e-8)
                for a, b in zip(p.probabilities, scaled)
            ):
                raise ValueError("probabilities do not match current logits")
        if (
            s is not None
            and s.eligible_owner is not None
            and (
                c is None
                or c.state != "predicted"
                or c.logits.index(max(c.logits)) != 0
                or c.track_id != s.eligible_owner
                or p is None
                or p.rejection_status != "accepted"
            )
        ):
            raise ValueError("owner offer needs a visible fresh model result")
        if (
            s is not None
            and s.candidate is not None
            and int(s.candidate.source_timestamp_ns) > int(self.timestamp_ns)
        ):
            raise ValueError("candidate cannot come from the future")
        return self


class ShadowPacket(PreviewPacket):
    def encode(self):
        data = self.snapshot.model_dump_json().encode()
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("shadow metadata exceeds budget")
        return PREFIX.pack(b"HSP5", len(data), len(self.jpeg)) + data + self.jpeg


def read_shadow_packet(stream):
    return read_packet(
        stream,
        expected_magic=b"HSP5",
        snapshot_type=ShadowSnapshot,
        packet_type=ShadowPacket,
    )
