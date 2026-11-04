"""Offline evaluation commands. Never train neural weights or dispatch actions."""

from __future__ import annotations

import argparse
import zipfile
from importlib.resources import files
from pathlib import Path

from algorithm.bundle import digest, json_bytes, read_bounded, strict_json
from algorithm.evaluation.calibration import (
    fit_calibration,
    load_calibration,
    save_calibration,
)
from algorithm.evaluation.classification import evaluate_classification
from algorithm.evaluation.event_contracts import EventInput
from algorithm.evaluation.events import evaluate_events
from algorithm.evaluation.scores import load_scores
from shared.classes import vocabulary_for


def implementation_digest():
    root = files("algorithm.evaluation")
    return digest(
        json_bytes(
            {
                p.name: digest(p.read_bytes())
                for p in sorted(root.iterdir(), key=lambda p: p.name)
                if p.name.endswith(".py")
            }
        )
    )


def classification_report(scores, calibration=None):
    provenance = scores.provenance
    report = {
        "format_version": "classification-report-v1",
        "synthetic": provenance.synthetic,
        "source": provenance.model_dump(mode="json"),
        "score_manifest_sha256": scores.manifest_sha256,
        "implementation_sha256": implementation_digest(),
        "actions_enabled": False,
        "classes": list(vocabulary_for(provenance.vocabulary_version).tokens),
        "scope": "fixed cache-represented endpoints, NOT entire recording timeline or online availability",
        "baseline": evaluate_classification(scores),
        "calibration": None,
        "limitations": [
            "classification is not command execution or out-of-distribution detection",
            "zero-support classes remain in macro denominators",
            "test reports do not authorize tuning or model selection",
        ],
    }
    if calibration is not None:
        report["calibration"] = calibration.model_dump(mode="json")
        report["temperature_scaled"] = evaluate_classification(
            scores, temperature=calibration.temperature
        )
        report["selective"] = (
            evaluate_classification(
                scores,
                temperature=calibration.temperature,
                threshold=calibration.threshold,
            )
            if calibration.rejection_status == "selected"
            else None
        )
        report["limitations"].append(
            "draft validation selection; empirical risk is not a population guarantee"
        )
    return report


def _write_report(path, report):
    data = json_bytes(report)
    with Path(path).open("xb") as stream:
        stream.write(data)
    return {
        "status": "report_written",
        "report_sha256": digest(data),
        "synthetic": report["synthetic"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser(
        "score", help="read-only CPU inference over a pinned val/test cache"
    )
    score.add_argument("--package", required=True)
    score.add_argument("--model-sha256", required=True)
    score.add_argument("--cache", required=True)
    score.add_argument(
        "--cache-index-sha256",
        required=True,
        help="canonical index_sha256 from cache build receipt",
    )
    score.add_argument("--split", choices=("val", "test"), required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--batch-size", type=int, default=256)
    score.add_argument("--threads", type=int, default=1)
    score.add_argument("--allow-synthetic", action="store_true")
    for name in ("classify", "calibrate", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument("--scores", required=True)
        command.add_argument(
            "--sha256",
            required=True,
            help="trusted score manifest SHA256 from score receipt",
        )
        command.add_argument("--output", required=True)
        if name in {"classify", "diagnose"}:
            command.add_argument("--calibration")
            command.add_argument("--calibration-sha256")
            if name == "diagnose":
                command.add_argument("--allow-synthetic", action="store_true")
                command.add_argument(
                    "--cache",
                    help="optional matching compact validation cache for history/readiness error subsets",
                )
        else:
            command.add_argument("--temperatures", type=float, nargs="+")
            command.add_argument("--thresholds", type=float, nargs="+")
            command.add_argument("--max-risk", type=float, default=0.1)
            command.add_argument("--min-coverage", type=float, default=0.5)
            command.add_argument("--min-accepted", type=int, default=20)
            command.add_argument("--allow-synthetic", action="store_true")
    events = commands.add_parser(
        "events",
        help="evaluate independent target/observed events and protocol time intervals",
    )
    events.add_argument("--input", required=True)
    events.add_argument(
        "--sha256", required=True, help="trusted event-input file SHA256"
    )
    events.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if Path(args.output).exists():
            raise FileExistsError("evaluation outputs are immutable; select a new path")
        if args.command == "score":
            from algorithm.evaluation.collect import collect_scores

            result = collect_scores(
                args.package,
                args.cache,
                args.output,
                model_sha256=args.model_sha256,
                cache_index_sha256=args.cache_index_sha256,
                split=args.split,
                batch_size=args.batch_size,
                threads=args.threads,
                allow_synthetic=args.allow_synthetic,
            )
        elif args.command == "events":
            raw = read_bounded(Path(args.input), 16 * 1024**2)
            if digest(raw) != args.sha256:
                raise ValueError("event input SHA256 mismatch")
            data = EventInput.model_validate_json(json_bytes(strict_json(raw)))
            report = evaluate_events(data) | {
                "input_sha256": args.sha256,
                "implementation_sha256": implementation_digest(),
            }
            result = _write_report(args.output, report)
        else:
            scores = load_scores(args.scores, expected_manifest_sha256=args.sha256)
            if args.command in {"classify", "diagnose"}:
                if bool(args.calibration) != bool(args.calibration_sha256):
                    raise ValueError(
                        "calibration requires both artifact path and trusted SHA256"
                    )
                calibration = (
                    load_calibration(
                        args.calibration,
                        expected_sha256=args.calibration_sha256,
                        target=scores.provenance,
                    )
                    if args.calibration
                    else None
                )
                if args.command == "diagnose":
                    from algorithm.evaluation.diagnostics import diagnose_validation

                    report = diagnose_validation(
                        scores,
                        calibration,
                        allow_synthetic=args.allow_synthetic,
                        cache=args.cache,
                    )
                    report["implementation_sha256"] = implementation_digest()
                else:
                    report = classification_report(scores, calibration)
                report["calibration_sha256"] = args.calibration_sha256
                result = _write_report(args.output, report)
            else:
                options = {
                    key: tuple(getattr(args, key))
                    for key in ("temperatures", "thresholds")
                    if getattr(args, key) is not None
                }
                value = fit_calibration(
                    scores,
                    allow_synthetic=args.allow_synthetic,
                    max_risk=args.max_risk,
                    min_coverage=args.min_coverage,
                    min_accepted=args.min_accepted,
                    **options,
                )
                destination = Path(args.output)
                destination.mkdir(parents=True, exist_ok=False)
                sha = save_calibration(value, destination / "calibration.json")
                report = classification_report(scores, value)
                report["calibration_sha256"] = sha
                report["selection"] = {
                    "split": "val",
                    "temperature_trials": len(value.temperatures),
                    "threshold_trials": len(value.thresholds),
                    "checkpoint_calibration_reuse": "same validation inventory; uncertainty not estimated",
                    "event_search_trials": 0,
                }
                receipt = _write_report(destination / "report.json", report)
                # No invented on/off/hold/cooldown values. T28 must search with the full engine.
                policy = (
                    "format_version: event-thresholds-draft-v1\nstatus: draft\nactions_enabled: false\n"
                    "selection_status: not_searched\nparameters: null\n"
                    f"calibration_sha256: {sha}\n"
                    "reason: requires_T23_integration_and_T28_validation_stream_search\n"
                ).encode()
                with (destination / "event_thresholds.yaml").open("xb") as stream:
                    stream.write(policy)
                result = {
                    "status": "draft_calibration_written",
                    "calibration_sha256": sha,
                    "report_sha256": receipt["report_sha256"],
                    "event_thresholds_sha256": digest(policy),
                    "synthetic": value.synthetic,
                    "rejection_status": value.rejection_status,
                    "actions_enabled": False,
                }
                with (destination / "receipt.json").open("xb") as stream:
                    stream.write(json_bytes(result))
        print(json_bytes(result).decode(), end="")
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
        parser.exit(2, f"evaluation rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
