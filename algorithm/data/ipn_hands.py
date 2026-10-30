"""IPN HandS source adapter: versioned labels, exact frame alignment, no relabeling.

This module understands the HandS fps=30 timeline, not the original IPN Hand
resized/deduplicated frames. It does not infer subjects or application intent.
"""

import csv
from dataclasses import dataclass
import io
import math
import re

from algorithm.data.source_archive import VerifiedArchive
from shared.classes import IPN_HANDS_VERSION, vocabulary_for


@dataclass(frozen=True)
class Segment:
    class_id: int
    start: int
    end: int  # source end is inclusive


@dataclass(frozen=True)
class VideoLabels:
    video: str
    source_split: str
    frame_count: int
    segments: tuple[Segment, ...]


def _integer(value, minimum=0):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError("expected an unsigned decimal source integer")
    result = int(value)
    if result < minimum:
        raise ValueError("source integer below minimum")
    return result


def _video_id(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_#.-]+", value)
        or ".." in value
    ):
        raise ValueError("invalid source video identifier")
    return value


def read_annotations(archive: VerifiedArchive, *, expected_videos=200):
    if type(expected_videos) is not int or expected_videos <= 0:
        raise ValueError("expected_videos must be a positive integer")
    vocabulary = vocabulary_for(IPN_HANDS_VERSION)
    classes = list(
        csv.DictReader(io.StringIO(archive.read_text(archive.find("classes.txt"))))
    )
    if len(classes) != vocabulary.class_count or any(
        set(row) != {"id", "code", "gesture"}
        or _integer(row["id"]) != index
        or row["code"] != vocabulary.tokens[index]
        or not row["gesture"].strip()
        for index, row in enumerate(classes)
    ):
        raise ValueError(
            "HandS source class table mismatch; do not use original IPN Hand labels"
        )
    splits = {}
    for split in ("train", "test"):
        names = archive.read_text(archive.find(f"splits/{split}_list.txt")).splitlines()
        if not names:
            raise ValueError("source split list is empty")
        for name in names:
            name = _video_id(name.strip())
            if name in splits:
                raise ValueError("duplicate or overlapping source split video")
            splits[name] = split
    if len(splits) != expected_videos:
        raise ValueError("source video count mismatch")
    counts = list(
        csv.DictReader(io.StringIO(archive.read_text(archive.find("frame_counts.csv"))))
    )
    videos = {}
    for row in counts:
        if set(row) != {"video", "set", "frames_ipnhands", "frames_ipnhand_original"}:
            raise ValueError("unexpected frame-count fields")
        video = _video_id(row["video"])
        if video in videos or row["set"] != splits.get(video):
            raise ValueError("frame-count/split mismatch or duplicate video")
        count = _integer(row["frames_ipnhands"], 1)
        _integer(row["frames_ipnhand_original"], 1)  # never used as the HandS timeline
        if count > 100_000:
            raise ValueError("unreasonable frames per video")
        segments = []
        next_frame = 1
        for line in csv.reader(
            io.StringIO(
                archive.read_text(archive.find(f"temporal_annotations/{video}.txt"))
            )
        ):
            if len(line) != 3:
                raise ValueError("temporal rows require class_id,start_frame,end_frame")
            label, start, end = (_integer(value.strip()) for value in line)
            if (
                label >= vocabulary.class_count
                or start != next_frame
                or end < start
                or end > count
            ):
                raise ValueError(
                    "invalid class/temporal gap, overlap or inclusive endpoint"
                )
            segments.append(Segment(label, start, end))
            next_frame = end + 1
        if next_frame != count + 1:
            raise ValueError("temporal annotations must cover every HandS frame")
        videos[video] = VideoLabels(video, row["set"], count, tuple(segments))
    if set(videos) != set(splits):
        raise ValueError("frame-count inventory differs from source split lists")
    temporal_members = [
        name
        for name in archive.members
        if name.endswith(".txt")
        and (
            name.startswith("temporal_annotations/") or "/temporal_annotations/" in name
        )
    ]
    if len(temporal_members) != expected_videos:
        raise ValueError("unexpected or missing temporal annotation files")
    return videos


def iter_landmark_frames(archive: VerifiedArchive, labels: VideoLabels):
    """Bounded to one source video's JSON; yields all frames in 1-based order."""
    data = archive.read_json(archive.find(labels.video + ".json"))
    if not isinstance(data, dict) or len(data) != labels.frame_count:
        raise ValueError("landmark frame count mismatch")
    segment_index = 0
    for index in range(1, labels.frame_count + 1):
        key = f"{labels.video}_{index:06d}.jpg"
        if key not in data:
            raise ValueError(
                "missing or misaligned landmark frame; no index repair permitted"
            )
        frame = data[key]
        if not isinstance(frame, dict) or "hand_landmarks" not in frame:
            raise ValueError("missing hand_landmarks field")
        hands = frame["hand_landmarks"]
        if not isinstance(hands, list) or len(hands) > 2:
            raise ValueError("HandS frame must contain zero to two hands")
        parsed = []
        for hand in hands:
            if not isinstance(hand, dict) or not isinstance(
                hand.get("landmarks"), dict
            ):
                raise ValueError("malformed hand landmark record")
            landmarks = hand["landmarks"]
            if set(landmarks) != {str(i) for i in range(21)}:
                raise ValueError("HandS requires exactly 21 MediaPipe-order points")
            points = [landmarks[str(i)] for i in range(21)]
            if any(
                not isinstance(point, list)
                or len(point) != 3
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in point
                )
                for point in points
            ):
                raise ValueError(
                    "HandS points must be finite numeric xyz; depth is not invented"
                )
            handedness = hand.get("handedness", "unknown")
            if handedness not in {"Left", "Right", "unknown"}:
                raise ValueError("unknown source handedness encoding")
            parsed.append({"landmarks": points, "handedness": handedness.lower()})
        while index > labels.segments[segment_index].end:
            segment_index += 1
        yield {
            "frame_id": index,
            "timestamp_ns": ((index - 1) * 1_000_000_000) // 30,
            "class_id": labels.segments[segment_index].class_id,
            "hands": parsed,
        }


def audit_annotations(path, *, expected_sha256, expected_videos=200):
    with VerifiedArchive(
        path,
        expected_sha256=expected_sha256,
        max_member_bytes=2 * 1024**2,
        max_total_bytes=32 * 1024**2,
    ) as archive:
        videos = read_annotations(archive, expected_videos=expected_videos)
        class_frames = [0] * vocabulary_for(IPN_HANDS_VERSION).class_count
        class_segments = [0] * len(class_frames)
        for video in videos.values():
            for segment in video.segments:
                class_frames[segment.class_id] += segment.end - segment.start + 1
                class_segments[segment.class_id] += 1
        return {
            "format_version": "ipn-hands-audit-v1",
            "status": "annotations_verified",
            "annotations_sha256": archive.digest,
            "vocabulary_version": IPN_HANDS_VERSION,
            "videos": len(videos),
            "source_split_videos": {
                s: sum(v.source_split == s for v in videos.values())
                for s in ("train", "test")
            },
            "frames": sum(v.frame_count for v in videos.values()),
            "class_frames": class_frames,
            "class_segments": class_segments,
            "timeline": "HandS original-video fps=30, 1-based frames, inclusive segment ends",
            "subjects_verified": False,
            "ready_for_training": False,
            "remaining": [
                "verify landmark archive",
                "review subject/session grouping",
                "freeze train/validation/test roles",
            ],
        }
