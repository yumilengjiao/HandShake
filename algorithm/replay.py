"""Integrity-pinned, streaming observation replay. Read-only logical time, no devices/actions."""

from __future__ import annotations

import argparse
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from algorithm.bundle import digest, json_bytes, read_bounded, strict_json
from algorithm.data.source_archive import file_sha256
from shared.contracts import FrameStatus, HandObservation, Ns, TransformMetadata
from shared.desktop_protocol import WireModel
from shared.observations import FrameInfo, ObservationFrame

MAX_EVENTS = 200_000
MAX_LINE = 262_144
MAX_BYTES = 512 * 1024**2


class ReplayFrame(WireModel):
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    transform: TransformMetadata
    status: FrameStatus
    observations: tuple[HandObservation, ...] = Field(max_length=16)

    def frame(self):
        s = self.status
        return ObservationFrame(
            FrameInfo(
                s.stream_id,
                s.source_frame_id,
                s.source_timestamp_ns,
                self.width,
                self.height,
                self.transform,
                s.timebase,
            ),
            s,
            self.observations,
        )

    @model_validator(mode="after")
    def consistent(self):
        self.frame()
        return self


class ReplayItem(WireModel):
    at_ns: Ns
    kind: Literal["start", "stop", "reset", "tick", "frame"]
    frame: ReplayFrame | None = None

    @model_validator(mode="after")
    def payload(self):
        if (self.kind == "frame") != (self.frame is not None):
            raise ValueError("frame payload required only for a frame event")
        if self.frame is not None and (
            self.frame.status.source_timestamp_ns > self.at_ns
            or self.frame.status.processed_at_ns is not None
            and self.frame.status.processed_at_ns > self.at_ns
        ):
            raise ValueError("frame cannot arrive before its source/perception time")
        return self


class ReplayManifest(WireModel):
    format_version: Literal["observation-replay-v1"]
    synthetic: bool
    review_reference: str = Field(min_length=1, max_length=256)
    stream_id: str = Field(min_length=1, max_length=128)
    timebase: Literal["synthetic", "recording_relative", "monotonic"]
    events: int = Field(ge=1, le=MAX_EVENTS)
    events_bytes: int = Field(ge=1, le=MAX_BYTES)
    events_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def source_label(self):
        if self.timebase == "synthetic" and not self.synthetic:
            raise ValueError("synthetic clock cannot be labelled real data")
        return self


class ReplayArchive:
    def __init__(self, directory, *, expected_sha256):
        root = Path(directory)
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise ValueError("trusted replay manifest SHA256 required")
        if root.is_symlink() or {p.name for p in root.iterdir()} != {
            "manifest.json",
            "events.jsonl",
        }:
            raise ValueError("replay archive inventory mismatch")
        raw = read_bounded(root / "manifest.json", 1_048_576)
        if digest(raw) != expected_sha256:
            raise ValueError("replay manifest SHA256 mismatch")
        self.manifest = ReplayManifest.model_validate_json(json_bytes(strict_json(raw)))
        self.sha256 = expected_sha256
        self.path = root / "events.jsonl"
        info = self.path.stat()
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != self.manifest.events_bytes
        ):
            raise ValueError("replay events must be a bounded regular file")
        self.stamp = self._stamp()
        if file_sha256(self.path) != self.manifest.events_sha256:
            raise ValueError("replay event hash mismatch")
        # First pass checks the COMPLETE timeline before inference or output creation.
        self.frames = sum(item.kind == "frame" for item in self.items())

    def _stamp(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

    def guard(self):
        if self.path.is_symlink() or self._stamp() != self.stamp:
            raise ValueError("immutable replay changed during use")

    def items(self):
        self.guard()
        previous = -1
        last_frame = last_source = -1
        count = 0
        enabled = False
        with self.path.open("rb") as stream:
            while raw := stream.readline(MAX_LINE + 1):
                if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
                    raise ValueError("replay line too large or truncated")
                count += 1
                if count > self.manifest.events:
                    raise ValueError("replay event count mismatch")
                item = ReplayItem.model_validate_json(json_bytes(strict_json(raw)))
                if item.at_ns < previous:
                    raise ValueError(
                        "replay timeline must be nondecreasing; file order breaks ties"
                    )
                previous = item.at_ns
                if item.kind in {"start", "reset", "stop"}:
                    if (
                        item.kind == "start"
                        and enabled
                        or item.kind == "reset"
                        and not enabled
                    ):
                        raise ValueError("invalid replay lifecycle transition")
                    enabled = item.kind != "stop"
                if item.frame is not None:
                    s = item.frame.status
                    if (s.stream_id, s.timebase) != (
                        self.manifest.stream_id,
                        self.manifest.timebase,
                    ):
                        raise ValueError("replay frame clock/stream mismatch")
                    if (
                        s.source_frame_id <= last_frame
                        or s.source_timestamp_ns <= last_source
                    ):
                        raise ValueError(
                            "replay frame identity/time must strictly increase"
                        )
                    last_frame, last_source = s.source_frame_id, s.source_timestamp_ns
                self.guard()
                yield item
        self.guard()
        if count != self.manifest.events:
            raise ValueError("replay event count mismatch")


def run_replay(
    directory,
    output,
    *,
    manifest_sha256,
    package,
    model_sha256,
    calibration=None,
    calibration_sha256=None,
    allow_synthetic=False,
    threads=1,
):
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("threads must be in [1,8]")
    if bool(calibration) != bool(calibration_sha256):
        raise ValueError("calibration path and SHA256 must be supplied together")
    root = Path(output)
    if root.exists():
        raise FileExistsError("replay outputs are immutable")
    archive = ReplayArchive(directory, expected_sha256=manifest_sha256)
    import numpy as np
    import torch

    from algorithm.bundle import load_bundle
    from algorithm.evaluation.calibration import load_for_bundle
    from algorithm.evaluation.classification import log_probabilities
    from algorithm.online import OnlineClassifier

    torch.set_num_threads(threads)
    bundle = load_bundle(package, expected_manifest_sha256=model_sha256)
    synthetic = archive.manifest.synthetic or bundle.synthetic_training
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic replay/model requires explicit opt-in")
    scaling = (
        load_for_bundle(calibration, expected_sha256=calibration_sha256, bundle=bundle)
        if calibration
        else None
    )
    now, epoch, enabled, deadline = 0, 0, False, None
    classifier = None
    counts, tokens = Counter(), Counter()
    root.mkdir(parents=True, exist_ok=False)
    with (root / "predictions.jsonl").open("xb") as stream:

        def emit(value):
            counts[value["kind"] + ":" + value.get("state", "")] += 1
            stream.write(json_bytes_compact(value))

        for index, item in enumerate(archive.items()):
            # Process the deadline BEFORE the next item, even across a long empty gap.
            if enabled and deadline is not None and deadline <= item.at_ns:
                now = deadline
                classifier.reset()
                emit(
                    {
                        "kind": "timeout",
                        "state": "input_stalled",
                        "at_ns": now,
                        "epoch": epoch,
                    }
                )
                deadline = None
            now = item.at_ns
            if item.kind in {"start", "stop", "reset"}:
                if (
                    item.kind == "start"
                    and enabled
                    or item.kind == "reset"
                    and not enabled
                ):
                    raise ValueError("invalid replay lifecycle transition")
                enabled = item.kind != "stop"
                epoch += 1
                classifier = (
                    OnlineClassifier(
                        bundle,
                        context_epoch=epoch,
                        clock=lambda: now,
                        timebase=archive.manifest.timebase,
                    )
                    if enabled
                    else None
                )
                deadline = None
                emit(
                    {
                        "kind": item.kind,
                        "at_ns": now,
                        "epoch": epoch,
                        "state": "preview" if enabled else "stopped",
                    }
                )
            elif item.kind == "frame":
                if not enabled:
                    emit(
                        {
                            "kind": "frame",
                            "state": "skipped_stopped",
                            "at_ns": now,
                            "epoch": epoch,
                            "source_frame_id": item.frame.status.source_frame_id,
                        }
                    )
                    continue
                notice = classifier.update(item.frame.frame())
                # Wall-clock model duration is deliberately excluded from deterministic logical replay.
                values = notice.model_dump(mode="json", exclude={"inference_ns"})
                values.update(
                    kind="frame",
                    at_ns=now,
                    timeline_index=index,
                    probabilities=[],
                    selective_status="unavailable",
                )
                if notice.state == "predicted":
                    token = bundle.vocabulary.tokens[int(np.argmax(notice.logits))]
                    tokens[token] += 1
                    if scaling is not None:
                        probabilities = np.exp(
                            log_probabilities([notice.logits], scaling.temperature)
                        )[0]
                        values["probabilities"] = probabilities.tolist()
                        values["calibrated"] = True
                        values["selective_status"] = (
                            "unavailable"
                            if scaling.threshold is None
                            else "accepted"
                            if probabilities.max() >= scaling.threshold
                            else "rejected"
                        )
                emit(values)
                source = item.frame.status.source_timestamp_ns
                deadline = source + bundle.normalizer.config.max_age_ns + 1
                if deadline <= now:  # stale arriving input is NOT a new lease
                    deadline = None
                    classifier.reset()
            else:
                emit(
                    {
                        "kind": "tick",
                        "at_ns": now,
                        "epoch": epoch,
                        "state": "preview" if enabled else "stopped",
                    }
                )
        if classifier is not None:
            classifier.reset()
        emit({"kind": "eof", "state": "stopped", "at_ns": now, "epoch": epoch + 1})
    archive.guard()
    summary = {
        "format_version": "observation-replay-report-v1",
        "synthetic": synthetic,
        "scope": "read_only_logical_classification_no_actions",
        "actions_enabled": False,
        "timing": "recorded arrival order and virtual deadlines; no wall-clock latency claims",
        "input_manifest_sha256": archive.sha256,
        "model_sha256": model_sha256,
        "calibration_sha256": calibration_sha256,
        "calibration_status": "draft" if scaling else "unavailable",
        "implementation_sha256": digest(Path(__file__).read_bytes()),
        "frames": archive.frames,
        "counts": dict(sorted(counts.items())),
        "predicted_class_counts": dict(sorted(tokens.items())),
        "predictions_sha256": file_sha256(root / "predictions.jsonl"),
        "complete": True,
    }
    (root / "report.json").write_bytes(json_bytes(summary))
    return summary


def json_bytes_compact(value):
    import json

    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--sha256", required=True)
        if name == "run":
            p.add_argument("--package", required=True)
            p.add_argument("--model-sha256", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--calibration")
            p.add_argument("--calibration-sha256")
            p.add_argument("--allow-synthetic", action="store_true")
            p.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            archive = ReplayArchive(args.input, expected_sha256=args.sha256)
            result = {
                "status": "verified",
                "frames": archive.frames,
                "source": archive.manifest.model_dump(mode="json"),
                "actions_enabled": False,
            }
        else:
            result = run_replay(
                args.input,
                args.output,
                manifest_sha256=args.sha256,
                package=args.package,
                model_sha256=args.model_sha256,
                calibration=args.calibration,
                calibration_sha256=args.calibration_sha256,
                allow_synthetic=args.allow_synthetic,
                threads=args.threads,
            )
        print(json_bytes(result).decode(), end="")
        return 0
    except (
        ValueError,
        OSError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
    ) as error:
        parser.exit(2, f"observation replay rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
