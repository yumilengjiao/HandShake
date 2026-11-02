"""Explicit synthetic scale probe; creates no real data and opens no devices.

Run from an installed checkout. Output must be a fresh directory, preferably on
a disk filesystem (not /tmp tmpfs). This is capacity evidence, never accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def _peak_bytes():
    """Return peak RSS in bytes, or None when the platform cannot expose it."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError, OSError):
        # Windows has no resource module. Missing optional telemetry must not
        # prevent argument validation or be reported as a fictitious zero RSS.
        return None
    return int(peak * (1 if platform.system() == "Darwin" else 1024))


def prepare(root, frames):
    from algorithm.data.source_archive import file_sha256
    from backend.importing.ipn_hands import import_ipn_hands
    from backend.perception.synthetic import skeleton
    from shared.classes import IPN_HANDS_TOKENS

    videos = [f"synthetic_scale_{role}" for role in ("train", "val", "test")]
    with zipfile.ZipFile(
        root / "annotations.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        z.writestr(
            "classes.txt",
            "id,code,gesture\n"
            + "".join(f"{i},{t},synthetic\n" for i, t in enumerate(IPN_HANDS_TOKENS)),
        )
        z.writestr("splits/train_list.txt", "\n".join(videos[:2]) + "\n")
        z.writestr("splits/test_list.txt", videos[2] + "\n")
        z.writestr(
            "frame_counts.csv",
            "video,set,frames_ipnhands,frames_ipnhand_original\n"
            + "".join(
                f"{video},{'test' if i == 2 else 'train'},{frames},{frames}\n"
                for i, video in enumerate(videos)
            ),
        )
        for video in videos:
            labels = []
            for label in range(15):
                start, end = label * frames // 15 + 1, (label + 1) * frames // 15
                labels.append(f"{label},{start},{end}\n")
            z.writestr(f"temporal_annotations/{video}.txt", "".join(labels))
    # Stream the diagnostic raw JSON instead of making three video dictionaries.
    with zipfile.ZipFile(
        root / "landmarks.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for ordinal, video in enumerate(videos):
            with z.open(video + ".json", "w") as output:
                output.write(b"{")
                for index in range(1, frames + 1):
                    points = skeleton(
                        0.45 + 0.02 * math.sin(index / 20) + ordinal * 0.0001
                    )
                    hand = {
                        "handedness": "Right",
                        "landmarks": {str(i): list(p) for i, p in enumerate(points)},
                    }
                    text = json.dumps(
                        {f"{video}_{index:06d}.jpg": {"hand_landmarks": [hand]}},
                        separators=(",", ":"),
                    )
                    output.write((("," if index > 1 else "") + text[1:-1]).encode())
                output.write(b"}")
    plan = {
        "format_version": "subject-plan-v1",
        "status": "reviewed",
        "reference": "procedural-synthetic-probe-only",
        "subject_splits": {
            f"synthetic-{role}": role for role in ("train", "val", "test")
        },
        "videos": {
            v: {"subject_id": f"synthetic-{role}", "session_id": v}
            for v, role in zip(videos, ("train", "val", "test"))
        },
    }
    (root / "subjects.json").write_text(json.dumps(plan))
    import_ipn_hands(
        root / "annotations.zip",
        root / "landmarks.zip",
        root / "subjects.json",
        root / "imported",
        annotations_sha256=file_sha256(root / "annotations.zip"),
        landmarks_sha256=file_sha256(root / "landmarks.zip"),
        expected_videos=3,
        synthetic=True,
    )


def measure(root, stage):
    manifest = json.loads((root / "imported/manifest.json").read_text())
    if manifest.get("synthetic") is not True:
        raise ValueError("this diagnostic only accepts explicitly synthetic imports")
    start = time.monotonic()
    if stage == "cache":
        from backend.importing.stream_features import build_stream_cache

        results = []
        for split in ("train", "val"):
            results.append(
                build_stream_cache(
                    root / "imported",
                    root / split,
                    split=split,
                    dataset_version="synthetic-capacity-v1",
                    review_reference="procedural-synthetic-probe-only",
                    stats=None if split == "train" else root / "train/stats.json",
                )
            )
        result = {"caches": results}
    else:
        for split in ("train", "val"):
            if (
                json.loads((root / split / "index.json").read_text()).get("synthetic")
                is not True
            ):
                raise ValueError("this diagnostic cannot train on a real cache")
        from algorithm.training.config import TrainConfig
        from algorithm.training.trainer import train

        result = train(
            root / "train",
            root / "val",
            root / "run",
            config=TrainConfig(model_type="tcn", max_epochs=1, batch_size=16),
        )
        with (root / "run/epochs.jsonl").open() as stream:
            epoch = json.loads(stream.readline())
        assert len(epoch["batches"]) <= 256 and epoch["batches_total"] > 256
        result = {
            "epochs": result["epochs"],
            "global_step": result["global_step"],
            "batch_details_retained": len(epoch["batches"]),
            "batches_total": epoch["batches_total"],
        }
    peak_rss = _peak_bytes()
    result.update(
        stage=stage,
        elapsed_seconds=time.monotonic() - start,
        peak_process_rss_bytes=peak_rss,
        peak_process_rss_source="resource.getrusage"
        if peak_rss is not None
        else "unavailable",
        synthetic=True,
        scope="capacity probe, not official dataset or recognition accuracy",
    )
    (root / f"{stage}-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=10001)
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="explicit opt-in on a training machine; default is cache-only",
    )
    parser.add_argument("--worker", choices=("cache", "train"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker:
        if args.worker == "train" and not args.with_training:
            parser.error("training requires explicit --with-training")
        measure(root, args.worker)
        return
    if not 10001 <= args.frames_per_video <= 25000:
        parser.error("the synthetic scale probe supports 10001..25000 frames per video")
    root.mkdir(parents=True, exist_ok=False)
    prepare(root, args.frames_per_video)
    for stage in ("cache", "train") if args.with_training else ("cache",):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--worker",
                stage,
                *(["--with-training"] if args.with_training else []),
            ],
            check=True,
            timeout=1200,
        )


if __name__ == "__main__":
    main()
