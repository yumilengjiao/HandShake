"""Verified IPN HandS archives -> bounded per-video T11 observation shards.

Subject/session identity must be reviewed explicitly, never guessed from a video
name. Source frame labels are retained separately from conservative per-track
supervision. All original archives remain untouched.
"""

from contextlib import contextmanager
from hashlib import sha256
import gzip
import io
import json
import os
from pathlib import Path

from algorithm.data.dataset import ObservationRecord, canonical_digest
from algorithm.data.ipn_hands import iter_landmark_frames, read_annotations
from algorithm.data.source_archive import VerifiedArchive, file_sha256, strict_json
from backend.perception.assembler import ObservationAssembler
from backend.perception.mediapipe_adapter import RawHand
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.contracts import TransformMetadata
from shared.observations import FrameInfo


def validate_subject_plan(plan, videos):
    if not isinstance(plan, dict) or set(plan) != {
        "format_version",
        "status",
        "reference",
        "subject_splits",
        "videos",
    }:
        raise ValueError("invalid reviewed subject plan fields")
    if (
        plan["format_version"] != "subject-plan-v1"
        or plan["status"] != "reviewed"
        or not isinstance(plan["reference"], str)
        or not plan["reference"].strip()
    ):
        raise ValueError(
            "subject/session grouping requires an explicit reviewed reference"
        )
    if not isinstance(plan["videos"], dict) or set(plan["videos"]) != set(videos):
        raise ValueError("subject plan must cover exactly the source videos")
    splits = plan["subject_splits"]
    if (
        not isinstance(splits, dict)
        or not splits
        or any(
            not isinstance(subject, str)
            or not subject
            or role not in {"train", "val", "test"}
            for subject, role in splits.items()
        )
    ):
        raise ValueError("invalid subject split roles")
    seen = set()
    for name, source in videos.items():
        item = plan["videos"][name]
        if (
            not isinstance(item, dict)
            or set(item) != {"subject_id", "session_id"}
            or any(not isinstance(value, str) or not value for value in item.values())
        ):
            raise ValueError("each video requires explicit subject_id and session_id")
        subject = item["subject_id"]
        if subject not in splits:
            raise ValueError("unknown subject in reviewed video mapping")
        role = splits[subject]
        if (source.source_split == "test") != (role == "test"):
            raise ValueError(
                "source test isolation/subject overlap violation; do not move test into development"
            )
        seen.add(subject)
    if seen != set(splits) or set(splits.values()) != {"train", "val", "test"}:
        raise ValueError(
            "require nonempty train/val/test subject groups without unused subjects"
        )
    return plan


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _gzip_writer(path):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                yield stream


def _line(stream, value):
    stream.write(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )


def import_ipn_hands(
    annotations,
    landmarks,
    subject_plan,
    output,
    *,
    annotations_sha256,
    landmarks_sha256,
    expected_videos=200,
    synthetic=False,
):
    """No output is created before hashes, label inventory and grouping pass.

    Later errors retain an explicitly incomplete manifest. Never auto-publish a
    frozen training split: T09/T10 must review coverage and technical exclusions.
    """
    root = Path(output)
    if root.exists():
        raise FileExistsError(
            "import output already exists; never overwrite a prior import"
        )
    with VerifiedArchive(
        annotations,
        expected_sha256=annotations_sha256,
        max_member_bytes=2 * 1024**2,
        max_total_bytes=32 * 1024**2,
    ) as ann:
        videos = read_annotations(ann, expected_videos=expected_videos)
    plan = validate_subject_plan(
        strict_json(Path(subject_plan).read_text(encoding="utf-8")), videos
    )
    with VerifiedArchive(landmarks, expected_sha256=landmarks_sha256) as source:
        # Reject ambiguous/missing/extra video JSONs before writing any derived data.
        members = {name: source.find(name + ".json") for name in videos}
        if set(members.values()) != {
            name for name in source.members if name.endswith(".json")
        }:
            raise ValueError(
                "landmark video inventory differs from annotation inventory"
            )
        root.mkdir(parents=True, exist_ok=False)
        (root / "recordings").mkdir()
        manifest = {
            "format_version": "ipn-hands-import-v1",
            "complete": False,
            "status": "importing",
            "synthetic": synthetic,
            "vocabulary_version": IPN_HANDS_VERSION,
            "classes_sha256": vocabulary_for(IPN_HANDS_VERSION).digest(),
            "annotations_sha256": annotations_sha256,
            "landmarks_sha256": landmarks_sha256,
            "subject_plan_sha256": canonical_digest(plan),
            "subject_plan": plan,
            "source": "IPN HandS MPHands; upstream SDK version/parameters not supplied",
            "timebase": "recording_relative",
            "timeline": "original-video fps=30 derived timeline; not exposure time",
            "sdk_completion_time": "unavailable_external",
            "license": "CC-BY-4.0",
            "supervision_policy": "single_visible_estimated_hand_only; all source frame labels separately retained",
            "context_available": False,
            "intent_available": False,
            "action_backend": "none",
            "ready_for_training": False,
            "recordings": [],
        }
        _atomic_json(root / "manifest.json", manifest)
        try:
            for video, labels in sorted(videos.items()):
                identity = plan["videos"][video]
                subject, session = identity["subject_id"], identity["session_id"]
                role = plan["subject_splits"][subject]
                member_sha256 = sha256(source.read_bytes(members[video])).hexdigest()
                name = sha256(video.encode()).hexdigest()[:24]
                records_path = root / "recordings" / f"{name}.records.jsonl.gz"
                frames_path = root / "recordings" / f"{name}.frames.jsonl.gz"
                assembler = ObservationAssembler()
                last_tracks = ()
                count = supervised = frames = ambiguous = empty = 0
                with (
                    _gzip_writer(records_path) as records_out,
                    _gzip_writer(frames_path) as frames_out,
                ):
                    for item in iter_landmark_frames(source, labels):
                        info = FrameInfo(
                            f"ipn:{video}",
                            item["frame_id"],
                            item["timestamp_ns"],
                            640,
                            480,
                            TransformMetadata(preview_mirrored=False),
                            "recording_relative",
                        )
                        row = assembler.assemble(
                            info, [RawHand(**hand) for hand in item["hands"]]
                        )
                        observations = {obs.track_id: obs for obs in row.observations}
                        tracks = (
                            tuple(observations)
                            or last_tracks
                            or ("unassigned-no-observation",)
                        )
                        single = len(observations) == 1
                        _line(
                            frames_out,
                            {
                                "frame_id": item["frame_id"],
                                "timestamp_ns": item["timestamp_ns"],
                                "source_class_id": item["class_id"],
                                "estimated_hands": len(observations),
                                "supervision_available": single,
                            },
                        )
                        for track in tracks:
                            record = ObservationRecord(
                                recording_id=video,
                                subject_id=subject,
                                session_id=session,
                                lineage_id=video,
                                original_sha256=member_sha256,
                                visible_subjects=[subject],
                                track_id=track,
                                context_epoch=0,
                                image_size=[640, 480],
                                frame_status=row.status,
                                observation=observations.get(track),
                                class_id=item["class_id"] if single else -1,
                                class_valid=single,
                                vocabulary_version=IPN_HANDS_VERSION,
                                source_split=role,
                                timebase="recording_relative",
                            )
                            _line(records_out, record.model_dump(mode="json"))
                            count += 1
                            supervised += int(record.class_valid)
                        if observations:
                            last_tracks = tuple(observations)
                        frames += 1
                        empty += int(not observations)
                        ambiguous += int(len(observations) > 1)
                manifest["recordings"].append(
                    {
                        "recording_id": video,
                        "subject_id": subject,
                        "session_id": session,
                        "split": role,
                        "original_landmark_member_sha256": member_sha256,
                        "frames": frames,
                        "records": count,
                        "supervised_records": supervised,
                        "empty_prediction_frames": empty,
                        "multiple_prediction_frames": ambiguous,
                        "records_path": str(records_path.relative_to(root)),
                        "records_sha256": file_sha256(records_path),
                        "frames_path": str(frames_path.relative_to(root)),
                        "frames_sha256": file_sha256(frames_path),
                    }
                )
                _atomic_json(root / "manifest.json", manifest)
            manifest.update(
                complete=True, status="imported_pending_quality_and_split_freeze"
            )
            _atomic_json(root / "manifest.json", manifest)
            return manifest
        except BaseException as error:
            manifest.update(
                status="incomplete", error=f"{type(error).__name__}: {error}"
            )
            _atomic_json(root / "manifest.json", manifest)
            raise


def iter_imported_records(directory):
    """Verify and stream committed observation shards; does not freeze T10 splits."""
    root = Path(directory).resolve()
    manifest = strict_json((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != "ipn-hands-import-v1"
        or manifest.get("complete") is not True
        or manifest.get("status") != "imported_pending_quality_and_split_freeze"
    ):
        raise ValueError("import is incomplete or has an unsupported format")
    if (
        manifest.get("vocabulary_version") != IPN_HANDS_VERSION
        or manifest.get("classes_sha256") != vocabulary_for(IPN_HANDS_VERSION).digest()
    ):
        raise ValueError("import vocabulary mismatch")
    plan = manifest["subject_plan"]
    if canonical_digest(plan) != manifest["subject_plan_sha256"]:
        raise ValueError("import subject plan hash mismatch")
    ids = [entry["recording_id"] for entry in manifest["recordings"]]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["videos"]):
        raise ValueError("import recording inventory mismatch")
    for entry in manifest["recordings"]:
        identity = plan["videos"][entry["recording_id"]]
        if (entry["subject_id"], entry["session_id"], entry["split"]) != (
            identity["subject_id"],
            identity["session_id"],
            plan["subject_splits"][identity["subject_id"]],
        ):
            raise ValueError("import recording subject/split mismatch")
        paths = {}
        for kind in ("records", "frames"):
            path = (root / entry[kind + "_path"]).resolve()
            if (
                not path.is_relative_to(root)
                or file_sha256(path) != entry[kind + "_sha256"]
            ):
                raise ValueError("import shard path/hash mismatch")
            paths[kind] = path
        count = 0
        with gzip.open(paths["records"], "rt", encoding="utf-8") as stream:
            while line := stream.readline(65537):
                if len(line) > 65536:
                    raise ValueError("import observation line limit exceeded")
                record = ObservationRecord.model_validate(strict_json(line))
                if (
                    record.recording_id,
                    record.subject_id,
                    record.session_id,
                    record.source_split,
                    record.original_sha256,
                ) != (
                    entry["recording_id"],
                    entry["subject_id"],
                    entry["session_id"],
                    entry["split"],
                    entry["original_landmark_member_sha256"],
                ) or record.vocabulary_version != IPN_HANDS_VERSION:
                    raise ValueError("import shard provenance mismatch")
                count += 1
                if count > entry["records"]:
                    raise ValueError("import shard record count mismatch")
                yield record
        if count != entry["records"]:
            raise ValueError("import shard record count mismatch")
