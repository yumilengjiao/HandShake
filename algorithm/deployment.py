"""Model-package commands. Explicit trusted-local export; no camera or actions."""

from __future__ import annotations

import argparse
import json
import platform
import time
import zipfile

from algorithm.bundle import export_bundle, load_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="export this trainer's trusted best checkpoint"
    )
    export.add_argument("--run", required=True)
    export.add_argument("--normalization", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-synthetic", action="store_true")
    for name in ("inspect", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--package", required=True)
        command.add_argument(
            "--sha256",
            required=True,
            help="trusted checksums.json SHA256 from export receipt",
        )
        command.add_argument("--threads", type=int, default=1)
        if name == "benchmark":
            command.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_bundle(
                args.run,
                args.normalization,
                args.output,
                allow_synthetic=args.allow_synthetic,
            )
        else:
            if not 1 <= args.threads <= 8:
                raise ValueError("threads must be in [1,8]")
            import torch

            torch.set_num_threads(args.threads)
            bundle = load_bundle(args.package, expected_manifest_sha256=args.sha256)
            result = {
                "model_version": bundle.model_version,
                "model_type": bundle.model_type,
                "vocabulary_version": bundle.vocabulary.vocabulary_version,
                "classes": list(bundle.vocabulary.tokens),
                "synthetic_training": bundle.synthetic_training,
                "calibrated": False,
                "actions_enabled": False,
                "feature_schema": bundle.normalizer.stats.schema_hash,
                "threads": args.threads,
            }
            if args.command == "benchmark":
                if not 1 <= args.iterations <= 10000:
                    raise ValueError("iterations must be in [1,10000]")
                import numpy as np

                values = torch.zeros(
                    (1, 94) if bundle.model_type == "mlp" else (1, 32, 94)
                )
                durations = []
                with torch.inference_mode():
                    for index in range(10 + args.iterations):
                        start = time.perf_counter_ns()
                        bundle.model(values)
                        if index >= 10:
                            durations.append((time.perf_counter_ns() - start) / 1e6)
                try:
                    import resource

                    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    peak_rss = int(
                        peak_rss * (1 if platform.system() == "Darwin" else 1024)
                    )
                except ImportError:
                    peak_rss = None
                result["benchmark"] = {
                    "input": "synthetic zeros, batch=1",
                    "device": "cpu",
                    "iterations": args.iterations,
                    "p50_ms": float(np.percentile(durations, 50)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "peak_process_rss_bytes": peak_rss,
                    "memory_scope": "whole process peak since startup, includes runtime/loader",
                    "environment": {
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "machine": platform.machine(),
                        "os": platform.system(),
                    },
                    "scope": "model forward only, NOT camera-to-action latency",
                }
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ImportError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(2, f"model package rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
