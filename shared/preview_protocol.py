"""Bounded binary preview v2. v1 synthetic NDJSON remains a separate endpoint."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from shared.desktop_protocol import Counter, SessionId, WireModel

MAGIC = b"HSP2"
PREFIX = struct.Struct(">4sII")
MAX_METADATA_BYTES = 16_384
MAX_JPEG_BYTES = 131_072


class PreviewCommand(WireModel):
    version: Literal[2]
    session_id: SessionId
    command_id: Counter
    epoch: Counter
    kind: Literal["start", "start_camera", "stop", "heartbeat", "shutdown"]


class PreviewPoint(WireModel):
    # MediaPipe can report landmarks slightly outside the image. Do not clamp them.
    x: float = Field(ge=-16, le=16, allow_inf_nan=False)
    y: float = Field(ge=-16, le=16, allow_inf_nan=False)
    z: float = Field(ge=-16, le=16, allow_inf_nan=False)


class PreviewHand(WireModel):
    landmarks: tuple[PreviewPoint, ...] = Field(min_length=21, max_length=21)
    validity: tuple[bool, ...] = Field(min_length=21, max_length=21)
    usable: bool


class PreviewSnapshot(WireModel):
    version: Literal[2] = 2
    session_id: SessionId
    sequence: Counter
    epoch: Counter
    last_command_id: Counter
    timestamp_ns: Counter
    state: Literal["stopped", "starting", "preview", "stopping"]
    reason: Literal[
        "initial",
        "synthetic_preview",
        "user_stop",
        "heartbeat_timeout",
        "stale_epoch",
        "protocol_error",
        "input_closed",
        "shutdown",
        "camera_starting",
        "camera_preview",
        "camera_unavailable",
        "perception_unavailable",
        "input_stalled",
        "preview_error",
    ]
    source: Literal["synthetic", "camera"]
    mode: Literal["dry_run"] = "dry_run"
    model_status: Literal["unavailable"] = "unavailable"  # gesture model, NOT MediaPipe
    control_available: Literal[False] = False
    frame_id: Counter
    frame_timestamp_ns: Counter | None
    frame_width: int = Field(ge=0, le=640)
    frame_height: int = Field(ge=0, le=480)
    timestamp_kind: Literal[
        "synthetic_generation", "software_read_complete_not_exposure"
    ]
    capture_hz: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    perception_status: Literal[
        "unavailable",
        "processed_hands",
        "processed_no_hand",
        "not_processed",
        "error",
    ]
    hands: tuple[PreviewHand, ...] = Field(max_length=2)

    @model_validator(mode="after")
    def consistent(self):
        if self.state != "preview":
            if (
                self.frame_timestamp_ns is not None
                or self.hands
                or self.frame_width
                or self.frame_height
            ):
                raise ValueError(
                    "inactive/transition snapshots cannot carry a current frame"
                )
        elif (
            self.frame_timestamp_ns is None
            or min(self.frame_width, self.frame_height) < 1
        ):
            raise ValueError("preview requires frame identity and dimensions")
        expected_clock = (
            "synthetic_generation"
            if self.source == "synthetic"
            else "software_read_complete_not_exposure"
        )
        if self.timestamp_kind != expected_clock:
            raise ValueError("source/time-kind mismatch")
        if self.hands and self.perception_status != "processed_hands":
            raise ValueError("hands without processed_hands status")
        return self


@dataclass(frozen=True)
class PreviewPacket:
    snapshot: PreviewSnapshot
    jpeg: bytes = b""

    def __post_init__(self):
        has_image = (
            self.snapshot.source == "camera" and self.snapshot.state == "preview"
        )
        if bool(self.jpeg) != has_image or len(self.jpeg) > MAX_JPEG_BYTES:
            raise ValueError("image/state mismatch or oversized image")
        if self.jpeg and not (
            self.jpeg.startswith(b"\xff\xd8") and self.jpeg.endswith(b"\xff\xd9")
        ):
            raise ValueError("expected JPEG framing")

    def encode(self):
        metadata = self.snapshot.model_dump_json().encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds limit")
        return PREFIX.pack(MAGIC, len(metadata), len(self.jpeg)) + metadata + self.jpeg


def read_packet(
    stream,
    *,
    expected_magic=MAGIC,
    snapshot_type=PreviewSnapshot,
    packet_type=PreviewPacket,
):
    """Blocking reader for tests/tools; lengths are checked before allocation."""

    def exact(length):
        chunks = bytearray()
        while len(chunks) < length:
            chunk = stream.read(length - len(chunks))
            if not chunk:
                raise EOFError("truncated preview packet")
            chunks.extend(chunk)
        return bytes(chunks)

    magic, metadata_size, image_size = PREFIX.unpack(exact(PREFIX.size))
    if (
        magic != expected_magic
        or not 0 < metadata_size <= MAX_METADATA_BYTES
        or image_size > MAX_JPEG_BYTES
    ):
        raise ValueError("unsupported/oversized preview framing")
    snapshot = snapshot_type.model_validate_json(exact(metadata_size))
    return packet_type(snapshot, exact(image_size))
