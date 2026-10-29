"""T06 recording codec. NPZ contains numeric/Unicode arrays only, never pickle.

A per-recording manifest is authoritative. Incomplete/cancelled recordings are
excluded by default; readers verify every committed chunk before yielding rows.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from pydantic import Field, StrictBool, model_validator
from typing import Literal

from shared.contracts import Contract, Identifier, Ns, TransformMetadata, HandObservation, FrameStatus
from shared.observations import FrameInfo, ObservationFrame

FORMAT_VERSION = "recording-v1"


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class CollectionEvent(Contract):
    timebase: Literal["recording_relative"] = "recording_relative"
    sequence: int = Field(strict=True, ge=0)
    timestamp_ns: Ns
    kind: Literal["start", "pause", "resume", "end", "cancel", "prompt", "retry", "health", "stop"]
    payload: dict = Field(default_factory=dict)


class ChunkEntry(Contract):
    path: str = Field(pattern=r"^chunks/[0-9]{6}\.npz$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frames: int = Field(strict=True, gt=0)
    first_ns: Ns
    last_ns: Ns
    raw_path: str | None = Field(default=None, pattern=r"^raw/[0-9]{6}\.npz$")
    raw_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    raw_frames: int = Field(default=0, strict=True, ge=0)

    @model_validator(mode="after")
    def check_chunk(self):
        if self.last_ns < self.first_ns:
            raise ValueError("invalid chunk time range")
        if (self.raw_path is None) != (self.raw_sha256 is None) or bool(self.raw_path) != bool(self.raw_frames):
            raise ValueError("raw asset/count/hash must appear together")
        return self


class RecordingManifest(Contract):
    timebase: Literal["recording_relative"] = "recording_relative"
    format_version: Literal[FORMAT_VERSION] = FORMAT_VERSION
    recording_id: str = Field(pattern=r"^[a-f0-9-]{36}$")
    subject_id: Identifier
    session_id: Identifier
    consent_reference: Identifier
    consent_keypoints: Literal[True]
    consent_raw: StrictBool = False
    synthetic: StrictBool = False
    created_utc: Identifier
    protocol_version: Identifier
    code_version: Identifier
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin_ns: Ns
    source_timebase: Literal["monotonic", "recording_relative", "synthetic"]
    source: dict
    perception: dict
    context_available: Literal[False] = False  # T17 interaction context is not fabricated
    complete: StrictBool = False
    state: Literal["incomplete", "complete", "cancelled"] = "incomplete"
    reason: str | None = None
    chunks: list[ChunkEntry] = Field(default_factory=list)
    events_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    frame_count: int = Field(default=0, strict=True, ge=0)
    stats: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_manifest(self):
        if self.complete != (self.state == "complete"):
            raise ValueError("complete/state disagree")
        if self.complete and (not self.chunks or not self.events_sha256):
            raise ValueError("complete recording needs committed frames and events")
        if len({c.path for c in self.chunks}) != len(self.chunks):
            raise ValueError("duplicate chunk paths")
        if self.frame_count != sum(c.frames for c in self.chunks):
            raise ValueError("manifest frame count mismatch")
        if not self.consent_raw and any(c.raw_path for c in self.chunks):
            raise ValueError("raw data without separate consent")
        return self


def relative_frame(frame, origin):
    info = frame.info
    if info.timestamp_ns < origin:
        raise ValueError("frame precedes recording origin")
    relative = FrameInfo(info.stream_id, info.frame_id, info.timestamp_ns - origin, info.width, info.height,
                         info.transform, "recording_relative")
    values = frame.status.model_dump()
    values.update(source_timestamp_ns=relative.timestamp_ns, timebase="recording_relative")
    if values["processed_at_ns"] is not None:
        values["processed_at_ns"] -= origin
    observations = tuple(HandObservation.model_validate(o.model_dump() | {
        "source_timestamp_ns": relative.timestamp_ns, "timebase": "recording_relative"}) for o in frame.observations)
    return ObservationFrame(relative, FrameStatus.model_validate(values), observations, frame.world_landmarks)


def save_chunk(path, frames):
    """Images are not accepted. Descriptor/observation metadata accompanies arrays."""
    descriptors, observations, indexes, worlds, present = [], [], [], [], []
    for i, frame in enumerate(frames):
        info = asdict(frame.info)
        info["transform"] = frame.info.transform.model_dump(mode="json")
        descriptors.append(json.dumps({"info": info, "status": frame.status.model_dump(mode="json")}, allow_nan=False))
        for j, obs in enumerate(frame.observations):
            observations.append(obs)
            indexes.append(i)
            world = frame.world_landmarks[j] if frame.world_landmarks else None
            worlds.append(world if world is not None else np.zeros((21,3)))
            present.append(world is not None)
    arrays = {
        "format_version": np.array(FORMAT_VERSION),
        "frames_json": np.asarray(descriptors, dtype=np.str_),
        "frame_id": np.array([f.info.frame_id for f in frames], dtype=np.int64),
        "time_ns": np.array([f.info.timestamp_ns for f in frames], dtype=np.int64),
        "frame_index": np.array(indexes, dtype=np.int64),
        "landmarks": np.asarray([o.landmarks_image for o in observations], dtype=np.float64).reshape(-1,21,3),
        "validity_mask": np.asarray([o.validity_mask for o in observations], dtype=np.bool_).reshape(-1,21),
        "observations_json": np.asarray([json.dumps(o.model_dump(mode="json", exclude={"landmarks_image", "validity_mask"}), allow_nan=False) for o in observations], dtype=np.str_),
        "world_landmarks": np.asarray(worlds, dtype=np.float64).reshape(-1,21,3),
        "world_present": np.asarray(present, dtype=np.bool_),
    }
    save_npz(path, arrays)


def save_npz(path, arrays):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("xb") as out:
            np.savez_compressed(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_chunk(path):
    with np.load(path, allow_pickle=False) as data:
        expected = {"format_version", "frames_json", "frame_id", "time_ns", "frame_index", "landmarks", "validity_mask", "observations_json", "world_landmarks", "world_present"}
        if set(data.files) != expected or str(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported NPZ fields/format")
        arrays = {name:data[name] for name in expected}
    f, n = len(arrays["frames_json"]), len(arrays["frame_index"])
    shapes = {"frame_id": (f,), "time_ns": (f,), "frame_index": (n,), "landmarks": (n,21,3),
              "validity_mask": (n,21), "observations_json": (n,), "world_landmarks": (n,21,3), "world_present": (n,)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"invalid {key} shape")
    for key in ("frame_id", "time_ns", "frame_index"):
        if arrays[key].dtype != np.dtype("int64") or (arrays[key] < 0).any():
            raise ValueError(f"invalid {key} integers")
    if arrays["validity_mask"].dtype != np.bool_ or arrays["world_present"].dtype != np.bool_:
        raise ValueError("invalid validity dtype")
    if (arrays["frame_index"] >= f).any() or not np.isfinite(arrays["landmarks"]).all() or not np.isfinite(arrays["world_landmarks"]).all():
        raise ValueError("invalid observation index/coordinates")
    groups = [[] for _ in range(f)]
    world_groups = [[] for _ in range(f)]
    for j in range(n):
        raw = json.loads(str(arrays["observations_json"][j]))
        obs = HandObservation.model_validate(raw | {"landmarks_image": arrays["landmarks"][j].tolist(), "validity_mask": arrays["validity_mask"][j].tolist()})
        i = int(arrays["frame_index"][j])
        groups[i].append(obs)
        world_groups[i].append(tuple(tuple(float(x) for x in p) for p in arrays["world_landmarks"][j]) if arrays["world_present"][j] else None)
    result = []
    for i, descriptor in enumerate(arrays["frames_json"]):
        raw = json.loads(str(descriptor))
        values = raw["info"]
        values["transform"] = TransformMetadata.model_validate(values["transform"])
        info = FrameInfo(**values)
        if (info.frame_id, info.timestamp_ns, info.timebase) != (int(arrays["frame_id"][i]), int(arrays["time_ns"][i]), "recording_relative"):
            raise ValueError("frame descriptor/array mismatch")
        result.append(ObservationFrame(info, FrameStatus.model_validate(raw["status"]), tuple(groups[i]), tuple(world_groups[i])))
    return result


def safe_asset(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("recording asset escapes root")
    return path


def load_manifest(root, *, allow_incomplete=False):
    manifest = RecordingManifest.model_validate_json((Path(root) / "manifest.json").read_text())
    if not allow_incomplete and not manifest.complete:
        raise ValueError("incomplete/cancelled recording excluded by default")
    return manifest


def read_events(root, *, allow_incomplete=False):
    manifest = load_manifest(root, allow_incomplete=allow_incomplete)
    path = safe_asset(root, "events.jsonl")
    if manifest.events_sha256 is not None and digest_file(path) != manifest.events_sha256:
        raise ValueError("events checksum mismatch")
    events = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            event = CollectionEvent.model_validate_json(line)
            if event.sequence != index:
                raise ValueError("events require contiguous stable sequence numbers")
            events.append(event)
    return events


def iter_recording(root, *, allow_incomplete=False):
    manifest = load_manifest(root, allow_incomplete=allow_incomplete)
    if manifest.complete:
        read_events(root)
    elif manifest.events_sha256 is not None and digest_file(safe_asset(root, "events.jsonl")) != manifest.events_sha256:
        raise ValueError("events checksum mismatch")
    previous = None
    for chunk in manifest.chunks:
        path = safe_asset(root, chunk.path)
        if digest_file(path) != chunk.sha256:
            raise ValueError("chunk checksum mismatch")
        if chunk.raw_path and digest_file(safe_asset(root, chunk.raw_path)) != chunk.raw_sha256:
            raise ValueError("raw checksum mismatch")
        frames = load_chunk(path)
        if len(frames) != chunk.frames or frames[0].info.timestamp_ns != chunk.first_ns or frames[-1].info.timestamp_ns != chunk.last_ns:
            raise ValueError("chunk frame count/time range mismatch")
        for frame in frames:
            if previous and (frame.info.stream_id != previous.stream_id or frame.info.frame_id <= previous.frame_id or frame.info.timestamp_ns <= previous.timestamp_ns):
                raise ValueError("recording source identity/time order mismatch")
            previous = frame.info
            yield frame


def recover_recording(root):
    """Validate committed chunks; write a NEW recovery report, never claim completeness."""
    manifest = load_manifest(root, allow_incomplete=True)
    if manifest.complete:
        raise ValueError("recording already complete; recovery is not an overwrite operation")
    count = sum(1 for _ in iter_recording(root, allow_incomplete=True))
    report = {"format_version": FORMAT_VERSION, "recording_id": manifest.recording_id,
              "complete": False, "verified_committed_frames": count,
              "reason": "uncommitted tail/events may be missing; manual review required"}
    target = Path(root) / f"recovery-{uuid4().hex}.json"
    atomic_json(target, report)
    return target
