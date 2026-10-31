"""Desktop prototype wire contract. No application-control capability in v1."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Decimal strings survive all language runtimes without IEEE-754 rounding.
Counter = Annotated[str, StringConstraints(pattern=r"^(0|[1-9][0-9]{0,18})$")]
SessionId = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{32}$")]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("version must be an integer, not a boolean")
        return value

    @field_validator("control_available", mode="before", check_fields=False)
    @classmethod
    def boolean_capability(cls, value):
        if type(value) is not bool:
            raise ValueError("capability must be a boolean")
        return value


class DesktopCommand(WireModel):
    version: Literal[1]
    session_id: SessionId
    command_id: Counter
    epoch: Counter
    kind: Literal["start", "stop", "heartbeat", "shutdown"]


class StagePoint(WireModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)


class DesktopSnapshot(WireModel):
    version: Literal[1] = 1
    session_id: SessionId
    sequence: Counter
    epoch: Counter
    last_command_id: Counter
    timestamp_ns: Counter
    state: Literal["stopped", "preview"]
    reason: Literal[
        "initial",
        "synthetic_preview",
        "user_stop",
        "heartbeat_timeout",
        "stale_epoch",
        "protocol_error",
        "input_closed",
        "shutdown",
    ]
    source: Literal["synthetic"] = "synthetic"
    mode: Literal["dry_run"] = "dry_run"
    model_status: Literal["unavailable"] = "unavailable"
    control_available: Literal[False] = False
    frame_id: Counter
    frame_timestamp_ns: Counter | None
    landmarks: tuple[StagePoint, ...]

    @field_validator("landmarks")
    @classmethod
    def skeleton_length(cls, value):
        if len(value) not in (0, 21):
            raise ValueError("a skeleton must contain exactly 21 points")
        return value

    @model_validator(mode="after")
    def frame_consistency(self):
        if self.state == "stopped" and (
            self.landmarks or self.frame_timestamp_ns is not None
        ):
            raise ValueError("stopped snapshots cannot contain a current frame")
        if self.state == "preview" and self.frame_timestamp_ns is None:
            raise ValueError("preview requires a source timestamp")
        return self
