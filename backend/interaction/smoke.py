"""Finite T17/T18 geometry/score fixture; no camera, model or system injection."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from backend.controller import ApplicationController
from backend.interaction.engine import Evidence, InteractionPolicy
from backend.interaction.runtime import InteractionRuntime
from shared.contracts import Gesture, ModelPrediction


def run_smoke():
    now = 0
    controller = ApplicationController("interaction-fixture", timebase="synthetic")
    runtime = InteractionRuntime(controller, model_version="synthetic-rule-fixture", clock=lambda: now, policy=replace(InteractionPolicy.defaults(), score_kind="rule_decision"), watchdog=False)
    report = dict(policy=asdict(runtime.engine.policy), config_sha256=runtime.mapper.config.digest(), synthetic=True, camera_opened=False, real_model_used=False, system_injection=False,
                  events=[], results=[], timeline=[])
    try:
        runtime.enable()
        def step(gesture, x=.5):
            nonlocal now
            now += 50_000_000
            p = ModelPrediction(stream_id="interaction-fixture", track_id="hand", context_epoch=controller.snapshot().context_epoch,
                                source_frame_id=now // 50_000_000, source_timestamp_ns=now, inference_timestamp_ns=now,
                                logits=[float(g == gesture) for g in Gesture], model_version="synthetic-rule-fixture", timebase="synthetic")
            e = Evidence(p, True, True, ("hand",), static_pose_valid=gesture in {Gesture.NO_ACTION, Gesture.WAKE_OPEN, Gesture.EXIT_POSE},
                         neutral_valid=gesture == Gesture.NO_ACTION, palm_xy=(x, .5))
            events, tickets = runtime.update(e)
            report["events"].extend(event.model_dump(mode="json") for event in events)
            report["results"].extend(ticket.future.result(1).model_dump(mode="json") for ticket in tickets)
            report["timeline"].append(runtime.engine.decisions[-1])
        for _ in range(13):
            step(Gesture.WAKE_OPEN)
        for _ in range(9):
            step(Gesture.NO_ACTION)
        for index, x in enumerate((.5, .57, .64, .71)):
            step(Gesture.SWIPE_RIGHT if index == 3 else Gesture.NO_ACTION, x)
        for x in (.64, .57, .5, .43, .36, .29):
            step(Gesture.SWIPE_LEFT, x)
        for _ in range(11):
            step(Gesture.NO_ACTION)
        for index, x in enumerate((.5, .43, .36, .29)):
            step(Gesture.SWIPE_LEFT if index == 3 else Gesture.NO_ACTION, x)
        for _ in range(18):
            step(Gesture.EXIT_POSE, .29)
        runtime.stop()
        released = runtime.dispatcher.wait_released()
        report.update(final_context=controller.snapshot().model_dump(mode="json"), safety=list(runtime.dispatcher.safety), released=released)
        assert [event["gesture_id"] for event in report["events"]] == ["wake_open", "swipe_right", "swipe_left", "exit_pose"]
        assert len(report["results"]) == 2 and all(result["injection_status"] == "not_attempted" for result in report["results"])
        return report
    finally:
        runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_smoke()
    if args.output:
        with args.output.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
