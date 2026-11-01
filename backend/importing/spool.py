"""Disk-backed audit/sort of verified IPN HandS import shards.

SQLite is a temporary spool, not a new authoritative dataset format. No whole
record list or source-ID set is retained in RAM; original files stay read-only.
"""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path

from algorithm.data.dataset import ObservationRecord, canonical_digest
from algorithm.data.source_archive import strict_json
from backend.importing.verified_lines import verified_json_lines
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.contracts import TransformMetadata

MAX_RECORDS = 10_000_000
MAX_GROUPS = 100_000


def read_manifest(root):
    with (Path(root) / "manifest.json").open("rb") as stream:
        data = stream.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("import manifest budget exceeded")
    manifest = strict_json(data)
    if (
        manifest.get("format_version") != "ipn-hands-import-v1"
        or manifest.get("complete") is not True
        or manifest.get("status") != "imported_pending_quality_and_split_freeze"
        or type(manifest.get("synthetic")) is not bool
        or manifest.get("vocabulary_version") != IPN_HANDS_VERSION
        or manifest.get("classes_sha256") != vocabulary_for(IPN_HANDS_VERSION).digest()
    ):
        raise ValueError("incomplete or unsupported import")
    plan = manifest["subject_plan"]
    if (
        plan.get("format_version") != "subject-plan-v1"
        or plan.get("status") != "reviewed"
        or not plan.get("reference")
        or canonical_digest(plan) != manifest["subject_plan_sha256"]
    ):
        raise ValueError("import subject plan mismatch")
    if set(plan["subject_splits"].values()) != {"train", "val", "test"}:
        raise ValueError("require three nonempty preassigned split roles")
    entries = manifest["recordings"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 10_000:
        raise ValueError("recording inventory budget")
    ids = [item["recording_id"] for item in entries]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["videos"]):
        raise ValueError("recording inventory mismatch")
    if set(plan["subject_splits"]) != {
        v["subject_id"] for v in plan["videos"].values()
    }:
        raise ValueError("unused or missing subject mapping")
    return manifest


def _member(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("import path escapes root")
    return path


class ImportSpool:
    def __init__(
        self, directory, database, *, max_records=2_000_000, max_groups=50_000
    ):
        if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
            raise ValueError("max_records outside bounded stream budget")
        if type(max_groups) is not int or not 1 <= max_groups <= MAX_GROUPS:
            raise ValueError("max_groups outside bounded stream budget")
        self.root = Path(directory).resolve()
        self.manifest = read_manifest(self.root)
        self.manifest_hash = canonical_digest(self.manifest)
        self.max_records, self.max_groups = max_records, max_groups
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA cache_size=-8192")  # 8 MiB SQLite page cache
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript("""
            CREATE TABLE groups (gkey TEXT PRIMARY KEY, role TEXT NOT NULL, identity TEXT NOT NULL);
            CREATE TABLE records (gkey TEXT, frame_id INTEGER, timestamp_ns INTEGER, body TEXT NOT NULL,
                                  PRIMARY KEY(gkey, frame_id));
            CREATE INDEX ordered_records ON records(gkey, timestamp_ns, frame_id);
            CREATE TABLE roles (kind TEXT, value TEXT, role TEXT NOT NULL, PRIMARY KEY(kind,value));
            CREATE TABLE frames (recording TEXT, frame_id INTEGER, class_id INTEGER,
                                 hands INTEGER, expected INTEGER, seen INTEGER DEFAULT 0,
                                 PRIMARY KEY(recording, frame_id));
        """)
        self.records = self.group_count = 0
        self.payload_hash = sha256()

    def close(self):
        self.connection.close()

    def _role(self, kind, value, role):
        self.connection.execute(
            "INSERT OR IGNORE INTO roles VALUES (?,?,?)", (kind, value, role)
        )
        prior = self.connection.execute(
            "SELECT role FROM roles WHERE kind=? AND value=?", (kind, value)
        ).fetchone()[0]
        if prior != role:
            raise ValueError(f"{kind} leaks across splits")

    def audit(self):
        plan = self.manifest["subject_plan"]
        declared = sum(entry["records"] for entry in self.manifest["recordings"])
        if declared > self.max_records:
            raise ValueError("import record budget exceeded")
        for entry in sorted(
            self.manifest["recordings"], key=lambda item: item["recording_id"]
        ):
            video = entry["recording_id"]
            identity = plan["videos"][video]
            subject, session = identity["subject_id"], identity["session_id"]
            role = plan["subject_splits"][subject]
            if (entry["subject_id"], entry["session_id"], entry["split"]) != (
                subject,
                session,
                role,
            ):
                raise ValueError("import recording grouping mismatch")
            for key in ("records", "frames"):
                if (
                    type(entry[key]) is not int
                    or not 1 <= entry[key] <= self.max_records
                ):
                    raise ValueError("invalid import counts")
            last_hands = 1
            for index, frame in enumerate(
                verified_json_lines(
                    _member(self.root, entry["frames_path"]),
                    entry["frames_sha256"],
                    max_lines=entry["frames"],
                ),
                1,
            ):
                if set(frame) != {
                    "frame_id",
                    "timestamp_ns",
                    "source_class_id",
                    "estimated_hands",
                    "supervision_available",
                }:
                    raise ValueError("invalid source frame ledger fields")
                if (
                    type(frame["frame_id"]) is not int
                    or frame["frame_id"] != index
                    or type(frame["timestamp_ns"]) is not int
                    or frame["timestamp_ns"] != (index - 1) * 1_000_000_000 // 30
                    or type(frame["source_class_id"]) is not int
                    or not 0 <= frame["source_class_id"] < 15
                    or type(frame["estimated_hands"]) is not int
                    or not 0 <= frame["estimated_hands"] <= 2
                    or frame["supervision_available"]
                    is not (frame["estimated_hands"] == 1)
                ):
                    raise ValueError("invalid source frame ledger semantics")
                hands = frame["estimated_hands"]
                expected = hands or last_hands
                if hands:
                    last_hands = hands
                self.connection.execute(
                    "INSERT INTO frames(recording,frame_id,class_id,hands,expected) VALUES (?,?,?,?,?)",
                    (video, index, frame["source_class_id"], hands, expected),
                )
            supervised = 0
            for value in verified_json_lines(
                _member(self.root, entry["records_path"]),
                entry["records_sha256"],
                max_lines=entry["records"],
            ):
                record = ObservationRecord.model_validate(value)
                status = record.frame_status
                if (
                    (
                        record.recording_id,
                        record.subject_id,
                        record.session_id,
                        record.source_split,
                        record.original_sha256,
                        record.lineage_id,
                        record.vocabulary_version,
                    )
                    != (
                        video,
                        subject,
                        session,
                        role,
                        entry["original_landmark_member_sha256"],
                        video,
                        IPN_HANDS_VERSION,
                    )
                    or record.visible_subjects != [subject]
                    or record.context_epoch != 0
                    or status.stream_id != f"ipn:{video}"
                    or record.timebase != "recording_relative"
                    or record.image_size != [640, 480]
                    or status.processing_time_origin != "unavailable_external"
                    or status.processed_at_ns is not None
                    or (
                        record.observation is not None
                        and record.observation.transform_metadata
                        != TransformMetadata(preview_mirrored=False)
                    )
                    or status.source_timestamp_ns
                    != (status.source_frame_id - 1) * 1_000_000_000 // 30
                ):
                    raise ValueError("import record identity/time/geometry mismatch")
                ledger = self.connection.execute(
                    "SELECT class_id,hands,expected,seen FROM frames WHERE recording=? AND frame_id=?",
                    (video, status.source_frame_id),
                ).fetchone()
                if ledger is None:
                    raise ValueError("record absent from source frame ledger")
                class_id, hands, expected, seen = ledger
                if (
                    seen >= expected
                    or record.class_valid != (hands == 1)
                    or record.class_id != (class_id if hands == 1 else -1)
                    or (record.observation is not None) != bool(hands)
                    or status.perception_status.value
                    != ("processed_hands" if hands else "processed_no_hand")
                ):
                    raise ValueError("source/record supervision or coverage mismatch")
                self.connection.execute(
                    "UPDATE frames SET seen=seen+1 WHERE recording=? AND frame_id=?",
                    (video, status.source_frame_id),
                )
                for kind, key in (
                    ("recording", video),
                    ("session", subject + "/" + session),
                    ("lineage", record.lineage_id),
                    ("original", record.original_sha256),
                ):
                    self._role(kind, key, role)
                gkey = json.dumps(
                    record.group_key, ensure_ascii=False, separators=(",", ":")
                )
                metadata = {
                    "recording_id": video,
                    "subject_id": subject,
                    "session_id": session,
                    "stream_id": status.stream_id,
                    "track_id": record.track_id,
                    "lineage_id": record.lineage_id,
                    "original_sha256": record.original_sha256,
                    "timebase": record.timebase,
                }
                inserted = self.connection.execute(
                    "INSERT OR IGNORE INTO groups VALUES (?,?,?)",
                    (gkey, role, json.dumps(metadata)),
                ).rowcount
                self.group_count += inserted
                if self.group_count > self.max_groups:
                    raise ValueError("group budget exceeded")
                body = record.model_dump_json()
                self.connection.execute(
                    "INSERT INTO records VALUES (?,?,?,?)",
                    (gkey, status.source_frame_id, status.source_timestamp_ns, body),
                )
                self.payload_hash.update(body.encode() + b"\n")
                self.records += 1
                supervised += record.class_valid
            if self.connection.execute(
                "SELECT count(*) FROM frames WHERE recording=? AND expected!=seen",
                (video,),
            ).fetchone()[0]:
                raise ValueError("source frame coverage mismatch")
            if supervised != entry["supervised_records"]:
                raise ValueError("supervision count mismatch")
            self.connection.commit()
        if canonical_digest(read_manifest(self.root)) != self.manifest_hash:
            raise ValueError("import manifest changed during audit")
        self.dataset_hash = canonical_digest(
            {
                "format_version": "stream-records-v1",
                "import_manifest_sha256": self.manifest_hash,
                "record_stream_sha256": self.payload_hash.hexdigest(),
            }
        )
        return self

    def groups(self, split):
        for key, identity in self.connection.execute(
            "SELECT gkey,identity FROM groups WHERE role=? ORDER BY gkey", (split,)
        ):
            yield key, json.loads(identity)

    def group_records(self, key):
        for (body,) in self.connection.execute(
            "SELECT body FROM records WHERE gkey=? ORDER BY timestamp_ns,frame_id",
            (key,),
        ):
            yield ObservationRecord.model_validate_json(body)
