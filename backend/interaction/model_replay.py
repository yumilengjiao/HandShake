"""Pinned model-session logical replay: default shadow; optional SYNTHETIC local execution."""

import argparse
import zipfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from algorithm.bundle import digest, json_bytes
from algorithm.data.source_archive import file_sha256
from algorithm.replay import (
    ReplayArchive,
    ReplayItem,
    ReplayManifest,
    json_bytes_compact,
)
from backend.interaction.model_session import ModelPresentationSession
from backend.interaction.public_replay import ReplayClock
from shared.contracts import Index


class SessionItem(ReplayItem):
    kind: Literal[
        "start", "stop", "tick", "frame", "request_owner", "surface_ack", "heartbeat"
    ]
    track_id: str | None = Field(default=None, min_length=1, max_length=128)
    epoch: Index | None = None
    revision: Index | None = None

    @model_validator(mode="after")
    def controls(self):
        if (self.kind == "request_owner") != (self.track_id is not None):
            raise ValueError("owner request requires exactly one explicit track")
        if (self.kind in {"start", "request_owner", "surface_ack", "heartbeat"}) != (
            self.epoch is not None
        ):
            raise ValueError("context-based controls require their captured epoch")
        if (self.kind in {"surface_ack", "heartbeat"}) != (self.revision is not None):
            raise ValueError("surface/heartbeat require their captured revision")
        return self


class SessionManifest(ReplayManifest):
    format_version: Literal["model-session-replay-v1"]


class SessionArchive(ReplayArchive):
    manifest_type = SessionManifest
    item_type = SessionItem
    # These are REQUESTS; a start may be refused by surface/quiescence checks.
    strict_lifecycle = False


def run_session(
    directory,
    output,
    *,
    manifest_sha256,
    package,
    model_sha256,
    calibration_path,
    calibration_sha256,
    execute_synthetic=False,
    allow_synthetic=False,
    threads=1,
):
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("threads must be in [1,8]")
    if type(execute_synthetic) is not bool:
        raise ValueError("explicit synthetic execution boolean required")
    root = Path(output)
    if root.exists():
        raise FileExistsError("model session reports are immutable")
    archive = SessionArchive(directory, expected_sha256=manifest_sha256)
    if execute_synthetic and (
        not archive.manifest.synthetic or archive.manifest.timebase != "synthetic"
    ):
        raise ValueError("real observations can only run in shadow mode")
    import torch

    torch.set_num_threads(threads)
    clock = ReplayClock()
    session = ModelPresentationSession.load(
        package,
        model_sha256=model_sha256,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha256,
        clock=clock,
        stream_id=archive.manifest.stream_id,
        timebase=archive.manifest.timebase,
        source_synthetic=archive.manifest.synthetic,
        allow_synthetic=allow_synthetic,
        execution="synthetic_application_local" if execute_synthetic else "shadow",
        watchdog=False,
    )
    counts, pages = Counter(), [1]
    try:
        root.mkdir(parents=True, exist_ok=False)
        with (root / "trace.jsonl").open("xb") as stream:
            for index, item in enumerate(archive.items()):
                deadline = session.next_deadline_ns()
                if deadline is not None and deadline <= item.at_ns:
                    clock.now = deadline
                    session.host.tick()
                    stream.write(
                        json_bytes_compact(
                            {
                                "kind": "deadline",
                                "at_ns": deadline,
                                "reason": session.host.reason,
                            }
                        )
                    )
                    counts["deadline_expirations"] += 1
                clock.now = item.at_ns
                value = {"index": index, "kind": item.kind, "at_ns": clock.now}
                if item.kind == "frame":
                    result = session.process(item.frame.frame())
                    value.update(
                        reason=result.reason,
                        classification=result.notice.model_dump(
                            mode="json", exclude={"inference_ns"}
                        )
                        if result.notice
                        else None,
                        event=result.event.model_dump(mode="json")
                        if result.event
                        else None,
                        application_result=None,
                    )
                    if result.notice is not None:
                        counts["classification:" + result.notice.state] += 1
                    if result.event is not None:
                        counts["policy_candidates"] += 1
                    if result.ticket is not None:
                        outcome = result.ticket.future.result(timeout=2)
                        value["application_result"] = outcome.model_dump(mode="json")
                        counts[
                            "application_accepted"
                            if outcome.accepted
                            else "application_rejected"
                        ] += 1
                        pages.append(session.host.backend.page)
                elif item.kind == "stop":
                    session.stop("recorded_stop")
                    if not session.host.dispatcher.wait_released():
                        raise RuntimeError("recorded stop did not quiesce")
                    value["accepted"] = True
                elif item.kind == "start":
                    value["accepted"] = session.enable(epoch=item.epoch)
                elif item.kind == "request_owner":
                    value["accepted"] = session.request_owner(
                        item.track_id, epoch=item.epoch
                    )
                elif item.kind in {"surface_ack", "heartbeat"}:
                    current = session.snapshot()
                    method = (
                        session.host.acknowledge_surface
                        if item.kind == "surface_ack"
                        else session.host.heartbeat
                    )
                    value["accepted"] = method(
                        sequence=current["sequence"],
                        epoch=item.epoch,
                        revision=item.revision,
                    )
                else:
                    session.host.tick()
                if value.get("accepted") is False:
                    counts["rejected_controls"] += 1
                snapshot = session.snapshot()
                value.update(
                    epoch=snapshot["epoch"],
                    model_warming=snapshot["model_warming"],
                    policy_state=snapshot["state"],
                    page=snapshot["page"],
                    reason=value.get("reason", snapshot["bridge_reason"]),
                )
                stream.write(json_bytes_compact(value))
        archive.guard()
        session.stop("end_of_replay")
        if not session.host.dispatcher.wait_released():
            raise RuntimeError("EOF did not quiesce")
        final = session.snapshot()
        report = {
            "format_version": "model-session-report-v1",
            "complete": True,
            "synthetic": session.synthetic,
            "execution": session.host.execution,
            "input_manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
            "calibration_sha256": calibration_sha256,
            "policy_sha256": session.host.engine.policy_sha256,
            "implementation_sha256": digest(
                json_bytes(
                    {
                        p.name: digest(p.read_bytes())
                        for p in (
                            Path(__file__),
                            Path(__file__).with_name("model_session.py"),
                            Path(__file__).with_name("model_host.py"),
                        )
                    }
                )
            ),
            "frames": archive.frames,
            "counts": dict(counts),
            "page_sequence": pages,
            "trace_sha256": file_sha256(root / "trace.jsonl"),
            "final": final,
            "live_available": False,
            "external_control_available": False,
            "limitations": [
                "logical inference clock; not a real latency measurement",
                "recorded surface ACK is not native drawing evidence",
                "shadow never executes; application execution is synthetic-only",
                "draft policies and fixture models do not certify real recognition",
            ],
        }
        (root / "report.json").write_bytes(json_bytes(report))
        return report
    finally:
        session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "sha256",
        "package",
        "model-sha256",
        "calibration",
        "calibration-sha256",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute-synthetic", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = run_session(
            args.input,
            args.output,
            manifest_sha256=args.sha256,
            package=args.package,
            model_sha256=args.model_sha256,
            calibration_path=args.calibration,
            calibration_sha256=args.calibration_sha256,
            execute_synthetic=args.execute_synthetic,
            allow_synthetic=args.allow_synthetic,
            threads=args.threads,
        )
        print(
            json_bytes(
                {
                    k: report[k]
                    for k in (
                        "complete",
                        "execution",
                        "synthetic",
                        "counts",
                        "page_sequence",
                    )
                }
            ).decode(),
            end="",
        )
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
        parser.exit(2, f"model session rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
