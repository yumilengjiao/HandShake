"""Opt-in v3 camera preview carrying uncalibrated model telemetry; v2 stays intact."""

from typing import Literal

from pydantic import model_validator

from shared.inference_protocol import ClassificationNotice
from shared.preview_protocol import (
    MAX_METADATA_BYTES,
    PREFIX,
    PreviewCommand,
    PreviewPacket,
    PreviewSnapshot,
    read_packet,
)


class ModelPreviewCommand(PreviewCommand):
    version: Literal[3]


class ModelPreviewSnapshot(PreviewSnapshot):
    version: Literal[3] = 3
    model_status: Literal["unavailable", "loaded_uncalibrated", "error"] = "unavailable"
    classification: ClassificationNotice | None = None

    @model_validator(mode="after")
    def classification_matches_frame(self):
        value = self.classification
        if value is not None:
            if (
                self.state != "preview"
                or self.source != "camera"
                or self.model_status != "loaded_uncalibrated"
                or value.context_epoch != self.epoch
                or value.source_frame_id != self.frame_id
                or value.source_timestamp_ns != self.frame_timestamp_ns
            ):
                raise ValueError("classification must match this frame and epoch")
            if value.state == "predicted" and (
                self.perception_status != "processed_hands"
                or len(self.hands) != 1
                or not self.hands[0].usable
                or not all(self.hands[0].validity)
            ):
                raise ValueError("prediction requires a single valid observed hand")
        elif self.model_status == "loaded_uncalibrated":
            raise ValueError("loaded model requires an explicit current-frame status")
        return self


def read_model_packet(stream):
    return read_packet(
        stream,
        expected_magic=b"HSP3",
        snapshot_type=ModelPreviewSnapshot,
        packet_type=ModelPreviewPacket,
    )


class ModelPreviewPacket(PreviewPacket):
    def encode(self):
        metadata = self.snapshot.model_dump_json().encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds limit")
        return (
            PREFIX.pack(b"HSP3", len(metadata), len(self.jpeg)) + metadata + self.jpeg
        )
