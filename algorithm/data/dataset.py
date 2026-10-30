"""T11 draft observation adapter, fixed subject splits and shared feature windows.

This is not T06's final recording format or T10's completed audit. Real inputs
require a frozen upstream manifest; tests and smoke runs explicitly say synthetic.
"""

from hashlib import sha256
import json
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StrictBool, model_validator

from shared import PROTOCOL_VERSION
from shared.classes import vocabulary_for
from shared.config import ConfigModel
from shared.contracts import Contract, FrameStatus, HandObservation, Identifier, Index, PerceptionStatus
from shared.features import FeatureConfig, FeatureExtractor, FeatureNormalizer, FeatureWindowBuilder

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Split = Literal["train", "val", "test"]


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ObservationRecord(Contract):
    recording_id: Identifier
    subject_id: Identifier
    session_id: Identifier
    lineage_id: Identifier
    original_sha256: Digest
    visible_subjects: Annotated[list[Identifier], Field(min_length=1)]
    track_id: Identifier
    context_epoch: Index
    image_size: Annotated[list[Annotated[int, Field(strict=True, gt=0)]], Field(min_length=2, max_length=2)]
    frame_status: FrameStatus
    observation: HandObservation | None
    class_id: Annotated[int, Field(strict=True, ge=-1)]
    class_valid: StrictBool
    vocabulary_version: str = PROTOCOL_VERSION
    source_split: Split | None = None
    parent_records_sha256: Digest | None = None
    augmentation: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_record(self):
        if self.class_valid != (0 <= self.class_id < vocabulary_for(self.vocabulary_version).class_count):
            raise ValueError("valid class must be a vocabulary id; masked class must be -1")
        if not self.class_valid and self.class_id != -1:
            raise ValueError("masked class_id must be -1")
        if self.subject_id not in self.visible_subjects or len(set(self.visible_subjects)) != len(self.visible_subjects):
            raise ValueError("visible_subjects must uniquely include the tracked subject")
        if self.frame_status.timebase != self.timebase:
            raise ValueError("frame status and recording must use one timebase")
        if self.augmentation and (self.source_split != "train" or self.parent_records_sha256 is None):
            raise ValueError("augmented observations require train source and parent hash")
        if self.observation is not None:
            observation, status = self.observation, self.frame_status
            if (observation.stream_id, observation.source_frame_id, observation.source_timestamp_ns, observation.timebase) != (status.stream_id, status.source_frame_id, status.source_timestamp_ns, status.timebase):
                raise ValueError("observation must refer to the exact frame status source")
            if observation.track_id != self.track_id or status.perception_status != PerceptionStatus.HANDS:
                raise ValueError("observation requires matching track and processed_hands status")
        return self

    @property
    def group_key(self) -> tuple:
        return (self.recording_id, self.subject_id, self.session_id, self.frame_status.stream_id, self.track_id)


def records_digest(records: list[ObservationRecord]) -> str:
    payloads = []
    for record in records:
        payload = record.model_dump(mode="json")
        # Preserve historical v1 digests when the new fields have legacy defaults.
        if record.vocabulary_version == PROTOCOL_VERSION:
            payload.pop("vocabulary_version")
        if record.frame_status.processing_time_origin == "local_runtime":
            payload["frame_status"].pop("processing_time_origin")
        payloads.append(payload)
    ordered = sorted(payloads, key=lambda item: json.dumps(item, sort_keys=True))
    return canonical_digest(ordered)


class SplitManifest(ConfigModel):
    format_version: Literal["t11-split-adapter-v1"] = "t11-split-adapter-v1"
    dataset_version: Identifier
    dataset_sha256: Digest
    status: Literal["draft", "frozen"]
    synthetic: StrictBool
    preregistration_reference: Identifier
    subject_splits: dict[Identifier, Split]
    vocabulary_version: str = PROTOCOL_VERSION

    @property
    def vocabulary(self):
        return vocabulary_for(self.vocabulary_version)

    @model_validator(mode="after")
    def check_manifest(self):
        vocabulary_for(self.vocabulary_version)
        if not self.subject_splits:
            raise ValueError("a preassigned subject split map is required")
        if not self.synthetic and self.status != "frozen":
            raise ValueError("real statistics/data export requires the frozen T10 input")
        return self

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))

    def validate_records(self, records: list[ObservationRecord]) -> None:
        if records_digest(records) != self.dataset_sha256:
            raise ValueError("records do not match the frozen dataset hash")
        seen = set()
        roles = {}
        recording_metadata = {}
        for record in records:
            if record.vocabulary_version != self.vocabulary_version:
                raise ValueError("record and split manifest vocabulary mismatch")
            participants = set(record.visible_subjects)
            if not participants.issubset(self.subject_splits):
                raise ValueError("record contains a subject absent from the preassigned split")
            role_set = {self.subject_splits[subject] for subject in participants}
            if len(role_set) != 1:
                raise ValueError("visible participants cross splits; use a separate stress set")
            role = role_set.pop()
            if record.source_split is not None and record.source_split != role:
                raise ValueError("derived source_split conflicts with the preassigned participant role")
            for key in (("recording", record.recording_id), ("lineage", record.lineage_id), ("original", record.original_sha256)):
                if roles.setdefault(key, role) != role:
                    raise ValueError("recording/original/lineage leaks across subject splits")
            metadata = (record.subject_id, record.session_id, record.lineage_id, record.original_sha256)
            # Multiple tracked subjects in one recording are allowed only in the same split.
            prior = recording_metadata.setdefault(record.recording_id, metadata[1:])
            if prior != metadata[1:]:
                raise ValueError("one recording changes session/original/lineage metadata")
            source_key = record.group_key + (record.frame_status.source_frame_id,)
            if source_key in seen:
                raise ValueError("duplicate source frame within a recording/track")
            seen.add(source_key)


def _groups(records: list[ObservationRecord], manifest: SplitManifest, split: Split):
    if split not in {"train", "val", "test"}:
        raise ValueError("unknown split")
    manifest.validate_records(records)
    grouped = {}
    for record in records:
        if manifest.subject_splits[record.subject_id] == split:
            grouped.setdefault(record.group_key, []).append(record)
    for key in sorted(grouped):
        yield sorted(grouped[key], key=lambda row: (row.frame_status.source_timestamp_ns, row.frame_status.source_frame_id))


def _extract(record, extractor):
    status = record.frame_status
    arguments = dict(timestamp_ns=status.source_timestamp_ns, source_frame_id=status.source_frame_id,
                     stream_id=status.stream_id, track_id=record.track_id, context_epoch=record.context_epoch)
    if status.perception_status == PerceptionStatus.NOT_PROCESSED:
        extractor.unprocessed(status.source_timestamp_ns, stream_id=status.stream_id,
                              track_id=record.track_id, context_epoch=record.context_epoch)
        return None
    if record.observation is not None:
        return extractor.extract(record.observation, record.image_size, context_epoch=record.context_epoch)
    return extractor.missing(**arguments)


def fit_training_normalizer(records: list[ObservationRecord], manifest: SplitManifest,
                            config: FeatureConfig | None = None) -> FeatureNormalizer:
    config = config or FeatureConfig()
    samples = []
    for group in _groups(records, manifest, "train"):
        extractor = FeatureExtractor(config)
        for record in group:
            sample = _extract(record, extractor)
            if sample is not None:
                samples.append(sample)
    return FeatureNormalizer.fit(samples, dataset_hash=manifest.dataset_sha256,
                                 split_hash=manifest.digest(), split="train", config=config, vocabulary=manifest.vocabulary)


class WindowDataset:
    """Map-style NumPy dataset; PyTorch DataLoader can collate it without a wrapper.

    Labels and readiness remain separate. All processed endpoints are inventoried;
    no-hand frames before any actual track remain unavailable, never fake tracks.
    """

    def __init__(self, records: list[ObservationRecord], manifest: SplitManifest, *, split: Split,
                 normalizer: FeatureNormalizer, config: FeatureConfig | None = None, stride: int = 1):
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer fixed before evaluation")
        self.config = config or FeatureConfig()
        if (normalizer.stats.dataset_hash != manifest.dataset_sha256 or normalizer.stats.split_hash != manifest.digest()
                or normalizer.config.config_hash != self.config.config_hash):
            raise ValueError("normalizer dataset/split/config provenance mismatch")
        if normalizer.vocabulary != manifest.vocabulary:
            raise ValueError("normalizer and dataset vocabulary mismatch")
        self.manifest, self.split, self.stride = manifest, split, stride
        self.normalizer = normalizer
        self.records = tuple(records)  # Full timeline survives for later event evaluation.
        self.items, self.endpoint_inventory = [], []
        for group in _groups(records, manifest, split):
            extractor = FeatureExtractor(self.config)
            builder = FeatureWindowBuilder(self.config, normalizer=normalizer)
            processed_index = 0
            for record in group:
                status = record.frame_status
                sample = _extract(record, extractor)
                if status.perception_status == PerceptionStatus.NOT_PROCESSED:
                    builder.unprocessed(status.source_timestamp_ns, stream_id=status.stream_id,
                                        track_id=record.track_id, context_epoch=record.context_epoch)
                    continue
                selected = processed_index % stride == 0
                processed_index += 1
                window = builder.append(sample) if sample is not None else None
                source = dict(recording_id=record.recording_id, subject_id=record.subject_id,
                              session_id=record.session_id, stream_id=status.stream_id, track_id=record.track_id,
                              context_epoch=record.context_epoch, source_frame_id=status.source_frame_id,
                              timestamp_ns=status.source_timestamp_ns, timebase=record.timebase)
                self.endpoint_inventory.append(source | dict(selected=selected, represented=window is not None,
                                                              ready=False if window is None else window.ready))
                if not selected or window is None:
                    continue
                x = np.zeros((self.config.window_size, 94), dtype=np.float32)
                actual = np.asarray(window.features, dtype=np.float32)
                x[-len(actual):] = actual
                self.items.append(dict(x=x, y=np.int64(record.class_id),
                                       loss_masks={"class_valid": np.bool_(record.class_valid)},
                                       source=source, ready=window.ready, ready_reason=window.ready_reason,
                                       valid_length=len(actual)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return item | {"x": item["x"].copy(), "source": item["source"].copy(), "loss_masks": item["loss_masks"].copy()}

    def coverage(self) -> dict:
        endpoints = [item for item in self.endpoint_inventory if item["selected"]]
        ready = sum(item["ready"] for item in endpoints)
        return dict(selected_endpoints=len(endpoints), represented=len(self), ready=ready,
                    ready_coverage=ready / len(endpoints) if endpoints else 0.0,
                    unprocessed_frames=sum(record.frame_status.perception_status == PerceptionStatus.NOT_PROCESSED
                                           for record in self.records if self.manifest.subject_splits[record.subject_id] == self.split))
