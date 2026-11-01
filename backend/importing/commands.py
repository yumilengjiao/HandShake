"""Offline public-data CLI: verify, import, and build explicitly reviewed caches."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import zipfile

from algorithm.data.ipn_hands import audit_annotations
from algorithm.data.source_archive import file_sha256, strict_json
from backend.importing.features import build_import_cache
from backend.importing.ipn_hands import import_ipn_hands


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("audit-ipn", "import-ipn", "build-cache", "build-stream-cache"):
        command = commands.add_parser(name)
        command.add_argument(
            "--report",
            type=Path,
            help="exclusive JSON report, including failure reasons",
        )
        if name not in {"build-cache", "build-stream-cache"}:
            command.add_argument(
                "--catalog",
                type=Path,
                required=True,
                help="reviewed data/sources.json; no checksum override",
            )
            command.add_argument("--annotations", type=Path, required=True)
        if name == "import-ipn":
            command.add_argument("--landmarks", type=Path, required=True)
            command.add_argument(
                "--subjects",
                type=Path,
                required=True,
                help="reviewed subject-plan-v1; no filename guessing",
            )
            command.add_argument("--output", type=Path, required=True)
        elif name in {"build-cache", "build-stream-cache"}:
            command.add_argument("--imported", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--split", choices=["train", "val", "test"], required=True
            )
            command.add_argument("--dataset-version", required=True)
            command.add_argument(
                "--review-reference",
                required=True,
                help="actual quality/split review; not a magic consent string",
            )
            command.add_argument("--stats", type=Path)
            command.add_argument(
                "--max-records",
                type=int,
                default=10000 if name == "build-cache" else 2_000_000,
                help="record audit budget; streaming v3 uses a disk spool, not a whole-record list",
            )
            if name == "build-stream-cache":
                command.add_argument("--max-groups", type=int, default=50_000)
                command.add_argument("--stride", type=int, default=1)
    command = commands.add_parser(
        "inspect-cache",
        help="read-only compact cache transfer preflight; does not train",
    )
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument(
        "--index-sha256",
        required=True,
        help="trusted index_sha256 from the build receipt",
    )
    command.add_argument("--report", type=Path)
    return parser


def _run(args):
    if args.mode == "inspect-cache":
        from algorithm.data.compact import inspect_compact_cache

        return inspect_compact_cache(
            args.cache, expected_index_sha256=args.index_sha256
        )
    if args.mode == "build-stream-cache":
        from backend.importing.stream_features import build_stream_cache

        return build_stream_cache(
            args.imported,
            args.output,
            split=args.split,
            dataset_version=args.dataset_version,
            review_reference=args.review_reference,
            stats=args.stats,
            max_records=args.max_records,
            max_groups=args.max_groups,
            stride=args.stride,
        )
    if args.mode == "build-cache":
        return build_import_cache(
            args.imported,
            args.output,
            split=args.split,
            dataset_version=args.dataset_version,
            review_reference=args.review_reference,
            stats=args.stats,
            max_records=args.max_records,
        )
    catalog = strict_json(args.catalog.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "public-dataset-sources-v1":
        raise ValueError("unsupported source catalog")
    candidates = [item for item in catalog["datasets"] if item["id"] == "ipn-hands"]
    if len(candidates) != 1 or candidates[0]["license"]["id"] != "CC-BY-4.0":
        raise ValueError("expected one reviewed IPN HandS source and license")
    entries = candidates[0]["files"]
    hashes = {item["filename"]: item["expected_sha256"] for item in entries}
    if len(hashes) != len(entries):
        raise ValueError("duplicate source archive filename")
    annotation_sha = hashes["IPN_HandS_annotations.zip"]
    if args.mode == "audit-ipn":
        result = audit_annotations(args.annotations, expected_sha256=annotation_sha)
    else:
        manifest = import_ipn_hands(
            args.annotations,
            args.landmarks,
            args.subjects,
            args.output,
            annotations_sha256=annotation_sha,
            landmarks_sha256=hashes["IPN_HandS_landmarks_MPHands.zip"],
        )
        result = {
            key: manifest[key]
            for key in ("format_version", "status", "complete", "ready_for_training")
        }
        result["recordings"] = len(manifest["recordings"])
    result["catalog_sha256"] = file_sha256(args.catalog)
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.report.open("x", encoding="utf-8")
            if args.report
            else nullcontext() as report
        ):
            try:
                result = _run(args)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                zipfile.BadZipFile,
                EOFError,
            ) as error:
                result = {
                    "status": "blocked",
                    "ready_for_training": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            text = (
                json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            if report:
                report.write(text)
        print(text, end="")
        return 2 if result["status"] == "blocked" else 0
    except OSError as error:
        print(f"data report error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
