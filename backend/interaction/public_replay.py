"""Opt-in procedural public-gesture replay; no camera, neural training or OS input."""

import argparse
from importlib.resources import files
from pathlib import Path

import numpy as np

from algorithm.bundle import digest, json_bytes
from algorithm.evaluation.calibration import fit_calibration, save_calibration
from algorithm.evaluation.event_contracts import (
    EventInput,
    PredictedEvent,
    RecordingPlan,
    TargetEvent,
)
from algorithm.evaluation.events import evaluate_events
from algorithm.evaluation.scores import (
    ScoreProvenance,
    load_scores,
    score_dtype,
    write_scores,
)
from backend.interaction.public_presentation import SyntheticPublicPresentation
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.classification import SourceClassification
from shared.contracts import HandObservation, TransformMetadata
from shared.observations import FrameInfo, ObservationFrame


def implementation_digest():
    modules = (
        "backend.interaction.public_replay",
        "backend.interaction.public_policy",
        "backend.interaction.public_presentation",
        "backend.interaction.geometry",
        "backend.actions.dispatcher",
        "backend.actions.backends.presentation",
        "backend.controller",
        "shared.public_interaction",
        "shared.classification",
        "shared.contracts",
        "algorithm.evaluation.calibration",
        "algorithm.evaluation.classification",
        "algorithm.evaluation.events",
        "algorithm.evaluation.event_contracts",
        "algorithm.evaluation.scores",
    )
    return digest(
        json_bytes(
            {
                module: digest(
                    files(module.rsplit(".", 1)[0])
                    .joinpath(module.rsplit(".", 1)[1] + ".py")
                    .read_bytes()
                )
                for module in modules
            }
        )
    )


class ReplayClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def synthetic_calibration(directory):
    """Calibrate hand-authored correct logits, NOT weights or a trained model."""
    provenance = ScoreProvenance(
        **{
            name: digest(("synthetic-public-fixture:" + name).encode())
            for name in (
                "model_sha256",
                "dataset_sha256",
                "split_sha256",
                "stats_sha256",
                "config_sha256",
                "cache_index_sha256",
                "endpoint_sha256",
            )
        },
        vocabulary_version=IPN_HANDS_VERSION,
        split="val",
        data_synthetic=True,
        model_synthetic=True,
    )
    rows = np.zeros(15, dtype=score_dtype(15))
    rows["label"] = np.arange(15)
    rows["class_valid"] = rows["ready"] = rows["available"] = 1
    rows["logits"] = np.eye(15) * 10
    receipt = write_scores(Path(directory) / "scores", provenance, [rows])
    scores = load_scores(
        Path(directory) / "scores", expected_manifest_sha256=receipt["manifest_sha256"]
    )
    calibration = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.5,),
        max_risk=0.0,
        min_coverage=1.0,
        min_accepted=15,
    )
    sha = save_calibration(calibration, Path(directory) / "calibration.json")
    return calibration, sha


def synthetic_input(
    context,
    *,
    timestamp_ns,
    frame_id,
    model_version,
    token="NOG",
    x=0.5,
    y=0.5,
    track="operator",
    mirrored=True,
    quality=1.0,
):
    """Procedural one-extended/three-folded hand, untranslated model axes.

    Its rigid translations test geometry/code only, not the real appearance or
    official semantics of IPN HandS Throw left/right.
    """
    points = np.zeros((21, 3))
    points[0, 1] = 0.08
    for index, mcp in enumerate((5, 9, 13, 17)):
        points[mcp : mcp + 4, 0] = (index - 1.5) * 0.04
        points[mcp : mcp + 4, 1] = (
            [0, -0.05, -0.09, -0.13] if index == 0 else [0, -0.05, -0.02, -0.01]
        )
    points[1:5, :2] = [[-0.04, 0.06], [-0.06, 0.04], [-0.08, 0.02], [-0.10, 0]]
    center = points[[0, 5, 9, 13, 17], :2].mean(0)
    points[:, :2] += np.array([x, y]) - center
    transform = TransformMetadata(preview_mirrored=mirrored)
    observation = HandObservation(
        stream_id=context.stream_id,
        source_frame_id=frame_id,
        source_timestamp_ns=timestamp_ns,
        timebase=context.timebase,
        track_id=track,
        landmarks_image=points.tolist(),
        validity_mask=[True] * 21,
        observation_valid=True,
        handedness="unknown",
        quality=quality,
        transform_metadata=transform,
    )
    info = FrameInfo(
        context.stream_id, frame_id, timestamp_ns, 640, 480, transform, context.timebase
    )
    frame = ObservationFrame(
        info,
        info.status("processed_hands", processed_at_ns=timestamp_ns),
        (observation,),
    )
    logits = [0.0] * 15
    logits[vocabulary_for(IPN_HANDS_VERSION).tokens.index(token)] = 10.0
    prediction = SourceClassification(
        stream_id=context.stream_id,
        source_frame_id=frame_id,
        source_timestamp_ns=timestamp_ns,
        timebase=context.timebase,
        track_id=track,
        context_epoch=context.context_epoch,
        inference_timestamp_ns=timestamp_ns,
        model_version=model_version,
        vocabulary_version=IPN_HANDS_VERSION,
        logits=logits,
    )
    return prediction, frame


def acknowledge(runtime):
    snapshot = runtime.snapshot()
    return runtime.acknowledge_surface(
        **{key: snapshot[key] for key in ("sequence", "epoch", "revision")}
    )


def authored_steps():
    """Targets are authored BEFORE running the engine, never inferred from outputs."""
    steps = [("NOG", 0.5, None)] * 10
    for index, token in enumerate(("TRT", "TRT", "TLT")):
        positions = (
            [0.5, 0.56, 0.62, 0.68] if token == "TRT" else [0.5, 0.44, 0.38, 0.32]
        )
        steps.extend(
            (
                token,
                x,
                ("next_slide" if token == "TRT" else "previous_slide")
                if j == 3
                else None,
            )
            for j, x in enumerate(positions)
        )
        if index < 2:
            steps.extend(
                ("TLT", x, None) for x in (0.62, 0.56, 0.5, 0.5)
            )  # return motion, not an intended second command
            steps.extend([("NOG", 0.5, None)] * 12)
    steps.extend([("TLT", 0.32, None)] * 10)  # holding class never retriggers
    return steps


def run_replay(output, *, allow_synthetic=False):
    if allow_synthetic is not True:
        raise ValueError("synthetic replay requires --allow-synthetic")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    calibration, sha = synthetic_calibration(root)
    clock = ReplayClock()
    runtime = SyntheticPublicPresentation(
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    try:
        if not acknowledge(runtime) or not runtime.enable():
            raise RuntimeError("synthetic initial surface/enable rejected")
        p, frame = synthetic_input(
            runtime.controller.snapshot(),
            timestamp_ns=0,
            frame_id=0,
            model_version=runtime.engine.model_version,
        )
        if not runtime.arm_owner("operator", p, frame, ready=True) or not acknowledge(
            runtime
        ):
            raise RuntimeError("synthetic owner binding/surface rejected")
        context = runtime.controller.snapshot()
        targets, predictions = [], []
        step_ns = 50_000_000
        steps = authored_steps()
        for i, (_, _, action) in enumerate(steps, 1):
            if action:
                targets.append(
                    TargetEvent(
                        event_id=f"authored-{i}",
                        recording_id=context.stream_id,
                        timebase="synthetic",
                        target_context=context.target_context,
                        mode="presentation",
                        kind="command",
                        action=action,
                        start_ns=(i - 3) * step_ns,
                        confirmation_ns=i * step_ns,
                    )
                )
        pages, trace, policy_events, action_results = [1], [], [], []
        active_intervals, unavailable_intervals = [], []
        for i, (token, x, _) in enumerate(steps, 1):
            previous_time = clock.now
            clock.now = i * step_ns
            if runtime.engine.state in {"ACTIVE", "CANDIDATE"}:
                active_intervals.append((previous_time, clock.now))
            elif runtime.engine.state == "SUSPENDED":
                unavailable_intervals.append((previous_time, clock.now))
            p, frame = synthetic_input(
                runtime.controller.snapshot(),
                timestamp_ns=clock.now,
                frame_id=i,
                model_version=runtime.engine.model_version,
                token=token,
                x=x,
            )
            event, ticket = runtime.update(p, frame, ready=True)
            if event is not None:
                policy_events.append(event.model_dump(mode="json"))
            if event is not None and ticket is not None:
                result = ticket.future.result(timeout=2)
                action_results.append(result.model_dump(mode="json"))
                predictions.append(
                    PredictedEvent(
                        event_id=event.event_id,
                        recording_id=context.stream_id,
                        timebase="synthetic",
                        target_context=context.target_context,
                        mode="presentation",
                        kind="command",
                        action=event.action,
                        scope="application-local",
                        source_ns=p.source_timestamp_ns,
                        prediction_ns=p.inference_timestamp_ns,
                        submitted_ns=event.commit_ns,
                        accepted_ns=event.commit_ns if result.accepted else None,
                        external_injection_status="not_attempted",
                        application_ns=result.completed_at_ns
                        if result.accepted
                        else None,
                    )
                )
                pages.append(runtime.backend.page)
            trace.append(
                {
                    "frame": i,
                    "source_ns": p.source_timestamp_ns,
                    "source_token": token,
                    "state": runtime.engine.state,
                    "reason": runtime.reason,
                    "page": runtime.backend.page,
                    "event_id": None if event is None else event.event_id,
                }
            )
            if not acknowledge(runtime):
                raise RuntimeError("synthetic surface receipt rejected")
        runtime.stop()
        if not runtime.dispatcher.wait_released():
            raise RuntimeError("synthetic backend did not confirm safe quiescence")
        end = clock.now + 1
        data = EventInput(
            format_version="event-evaluation-v1",
            synthetic=True,
            protocol_reference="preauthored-public-replay-v1",
            early_ns=0,
            late_ns=0,
            recordings=(
                RecordingPlan(
                    recording_id=context.stream_id,
                    timebase="synthetic",
                    start_ns=0,
                    end_ns=end,
                    valid_intervals=((0, end),),
                    non_operation_intervals=((0, end),),
                    enabled_intervals=((0, clock.now),),
                    active_intervals=tuple(active_intervals),
                    unavailable_intervals=tuple(unavailable_intervals),
                    denominator_reference="synthetic-authored-timeline-not-ACTIVE-derived",
                    validity_origin="protocol_or_recording_damage",
                ),
            ),
            targets=tuple(targets),
            predictions=tuple(predictions),
        )
        report = {
            "format_version": "public-policy-replay-v1",
            "synthetic": True,
            "scope": "application-local",
            "live_available": False,
            "external_control_available": False,
            "fixture": "hand-authored logits and rigid hand geometry; no trained model",
            "surface_evidence": "synthetic host acknowledgements, NOT Flutter drawing",
            "active_time_policy": "piecewise engine ACTIVE/CANDIDATE intervals; not used to crop truth/background",
            "model_version": runtime.engine.model_version,
            "calibration_sha256": sha,
            "policy_sha256": runtime.engine.policy_sha256,
            "implementation_sha256": implementation_digest(),
            "page_sequence": pages,
            "policy_events": policy_events,
            "commands": [
                command.model_dump(mode="json") for command in runtime.commands
            ],
            "action_results": action_results,
            "trace": trace,
            "events": evaluate_events(data),
            "final": runtime.snapshot(),
        }
        (root / "event_input.json").write_bytes(
            json_bytes(data.model_dump(mode="json"))
        )
        (root / "policy.json").write_bytes(
            json_bytes(runtime.engine.policy.model_dump(mode="json"))
        )
        raw = json_bytes(report)
        (root / "report.json").write_bytes(raw)
        receipt = {
            "format_version": "synthetic-public-replay-receipt-v1",
            "complete": True,
            "files": {
                name: {
                    "sha256": digest((root / name).read_bytes()),
                    "bytes": (root / name).stat().st_size,
                }
                for name in (
                    "report.json",
                    "calibration.json",
                    "policy.json",
                    "event_input.json",
                    "scores/manifest.json",
                    "scores/scores.bin",
                )
            },
            "status": "synthetic_replay_completed",
            "report_sha256": digest(raw),
            "page_sequence": pages,
            "synthetic": True,
            "live_available": False,
            "external_control_available": False,
        }
        (root / "receipt.json").write_bytes(json_bytes(receipt))
        return receipt
    finally:
        runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_replay(args.output, allow_synthetic=args.allow_synthetic)
        print(json_bytes(result).decode(), end="")
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError) as error:
        parser.exit(2, f"public replay rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
