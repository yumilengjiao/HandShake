"""Safe CLI: synthetic dry-run, capture/collection and offline annotation tools."""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import yaml

from backend.mock import run_synthetic
from shared.config import load_config
from shared.records import write_jsonl


def code_version() -> str:
    version = metadata.version("handshake-core")
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return version
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout)
        return f"{version}+{commit}" + (".dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return version


def main(argv=None) -> int:
    from backend.collection.commands import add_arguments, run_capture_command, run_data_command

    parser = argparse.ArgumentParser(description="HandShake: local capture/collection tools; live device control is unavailable")
    parser.add_argument("mode", nargs="?", default="dry_run", choices=["dry_run", "preview", "camera_probe", "collect", "inspect", "recover", "annotate", "train", "evaluate", "replay", "live"])
    parser.add_argument("--config-dir", type=Path, help="directory containing the six YAML files and platforms/mock.yaml")
    parser.add_argument("--log", type=Path, help="write versioned synthetic JSONL; parent must exist; never overwrites")
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.mode in {"train", "evaluate", "replay", "live"}:
        reason = {
            "train": "dataset/training/model is not implemented (T10-T14)",
            "evaluate": "trained model and evaluator are not implemented (T15)",
            "replay": "recording replay is not implemented (T23); JSONL validation alone is not replay",
            "live": "no real backend or independent stop has been verified (T18/T19/T24)",
        }[args.mode]
        print(f"unavailable: {reason}", file=sys.stderr)
        return 2
    try:
        if args.mode in {"inspect", "recover", "annotate"}:
            return run_data_command(args)
        config = load_config(args.config_dir)
        if args.mode in {"preview", "camera_probe", "collect"}:
            if args.log:
                raise ValueError("--log is for synthetic dry_run; use --report for capture tools")
            return run_capture_command(args, config, code_version=code_version())
        if config.runtime.mode != "dry_run":
            raise ValueError("this scaffold only runs runtime.mode=dry_run")
        records = run_synthetic(config, code_version=code_version())
        if args.log:
            write_jsonl(args.log, records)
        for record in records:
            print(record.model_dump_json())
    except (OSError, RuntimeError, ValueError, ValidationError, yaml.YAMLError) as error:
        print(f"configuration/run error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
