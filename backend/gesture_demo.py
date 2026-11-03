"""Experimental hand-gesture presentation, strictly inside this application's window.

Explicit CLI opt-in AND keyboard camera/enable/owner steps are required. The
existing Flutter shadow preview remains non-executing. Nothing injects OS input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


TITLE = "HandShake - LOCAL gesture presentation (experimental)"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--run",
        type=Path,
        required=True,
        help="explicitly trusted local handoff with export/calibration receipts",
    )
    cli.add_argument(
        "--allow-local-demo",
        action="store_true",
        help="opt in to experimental application-local page changes, not external input",
    )
    cli.add_argument(
        "--verify-only", action="store_true", help="verify without camera or UI"
    )
    cli.add_argument(
        "--perception-model", type=Path, help="local MediaPipe hand_landmarker.task"
    )
    cli.add_argument("--camera", type=int, default=0)
    cli.add_argument(
        "--slides",
        type=Path,
        help="1..32 local PNG/JPEG slides; numbered names recommended",
    )
    return cli


def run_ui(assets, *, perception_model, camera=0, slides=None):
    import cv2 as cv
    import numpy as np

    from backend.interaction.live_demo import local_demo_deck
    from backend.perception.local_demo import DemoCameraWorker

    deck, images = local_demo_deck(), None
    if slides is not None:
        from backend.demo_slides import load_slide_images

        deck, images = load_slide_images(slides)
    worker = None
    last_camera = None
    message = "C: open camera. No control is enabled at startup."
    status = 0
    cv.namedWindow(TITLE, cv.WINDOW_AUTOSIZE)

    def label(canvas, text, x, y, *, scale=0.55, color=(225, 225, 225)):
        cv.putText(
            canvas, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA
        )

    try:
        while True:
            adapter = worker.adapter if worker is not None else None
            snapshot = adapter.snapshot() if adapter is not None else None
            if worker is not None:
                packet = worker.take()
                if packet is not None:
                    last_camera = packet
                if worker.error or worker.done.is_set():
                    last_camera = None
                    if not worker.stop_requested.is_set():
                        worker.request_stop()
                    message = "Camera stopped/error. C can restart only after confirmed release."
            canvas = np.full((760, 1280, 3), 28, dtype=np.uint8)
            page = snapshot["page"] if snapshot else 1
            slide = deck[0].slides[page - 1]
            if images is not None:
                canvas[60:600, 20:980] = images[page - 1]
            else:
                cv.rectangle(canvas, (20, 60), (980, 600), (245, 245, 245), -1)
                label(canvas, slide.title, 65, 170, scale=1.3, color=(20, 20, 20))
                for i, line in enumerate(slide.lines):
                    label(
                        canvas, line, 65, 260 + i * 52, scale=0.63, color=(20, 20, 20)
                    )
            label(
                canvas,
                f"LOCAL SLIDES  {page}/{len(deck[0].slides)}  |  No OS/PPT keyboard injection",
                24,
                35,
            )
            if (
                last_camera is not None
                and 0 <= time.monotonic_ns() - last_camera.timestamp_ns <= 150_000_000
            ):
                image = cv.imdecode(
                    np.frombuffer(last_camera.jpeg, dtype=np.uint8), cv.IMREAD_COLOR
                )
                if image is not None:
                    canvas[60:255, 1000:1260] = cv.resize(image, (260, 195))
            else:
                label(canvas, "No fresh camera image", 1000, 100, scale=0.43)
            if snapshot:
                label(canvas, "State: " + snapshot["state"], 1000, 290, scale=0.48)
                label(
                    canvas,
                    "Owner: "
                    + (
                        "bound"
                        if adapter.session.controller.snapshot().owner_track_id
                        else "none"
                    ),
                    1000,
                    320,
                    scale=0.48,
                )
                label(
                    canvas,
                    "O available: " + str(bool(snapshot["eligible_owner"])),
                    1000,
                    350,
                    scale=0.48,
                )
                label(
                    canvas,
                    "Page authority: " + str(snapshot["actions_enabled"]),
                    1000,
                    380,
                    scale=0.43,
                )
                notice = snapshot["classification"]
                if notice and notice["state"] == "predicted":
                    logits = notice["logits"]
                    token = adapter.bundle.vocabulary.tokens[
                        max(range(len(logits)), key=logits.__getitem__)
                    ]
                    label(canvas, "Top class: " + token, 1000, 420, scale=0.48)
                label(
                    canvas,
                    "Policy: " + snapshot["bridge_reason"][:95],
                    24,
                    625,
                    scale=0.48,
                )
            label(canvas, message[:125], 24, 655, scale=0.48)
            label(
                canvas,
                "C camera | E enable | O choose neutral owner | SPACE stop control | ESC/Q close camera + exit",
                24,
                693,
                scale=0.53,
            )
            label(
                canvas,
                "After O: keep still for warmup/release. TLT=previous / TRT=next. Errors remain possible.",
                24,
                723,
                scale=0.52,
            )
            cv.imshow(TITLE, canvas)
            key = cv.waitKey(20) & 0xFF
            if cv.getWindowProperty(TITLE, cv.WND_PROP_VISIBLE) < 1:
                break
            # The receipt follows the local UI paint/event pump, never precedes it.
            # It is not a claim that the OS compositor or the human saw the page.
            if (
                snapshot is not None
                and not worker.stop_requested.is_set()
                and not adapter.acknowledge(snapshot)
            ):
                # An owner bind or page commit may legitimately advance the epoch
                # while this frame paints. Do not turn that race into an explicit
                # stop. Dispatch stays blocked until a fresh receipt; the host's
                # fixed render/heartbeat deadlines still revoke on a stalled UI.
                message = (
                    "Surface changed during paint; waiting for the next fresh page."
                )
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                if worker is not None and not (
                    worker.done.is_set() and not worker.thread.is_alive()
                ):
                    message = "Camera already running or release unconfirmed; no second worker."
                    continue
                worker = DemoCameraWorker(
                    allow_application_local=True,
                    deck=deck,
                    device=camera,
                    model_path=perception_model,
                    model_package=assets["package"],
                    model_sha256=assets["sha256"],
                    calibration_path=assets["calibration"],
                    calibration_sha256=assets["calibration_sha256"],
                )
                worker.start()
                last_camera = None
                message = "Camera starting; then E enables control. No automatic owner selection."
            elif key == ord(" ") and adapter is not None:
                adapter.stop()
                message = "Control STOPPED. Camera preview remains on. ESC closes it."
            elif key == ord("e") and snapshot is not None:
                ok = adapter.enable(snapshot)
                message = (
                    "Enabled: present one neutral hand. O becomes available after a full window."
                    if ok
                    else "Enable rejected (busy/stale/already enabled); try again when stopped."
                )
            elif key == ord("o") and snapshot is not None:
                ok = adapter.request_owner(snapshot)
                message = (
                    "Owner requested. Keep hand still for rewarm and release."
                    if ok
                    else "Owner rejected: one fresh neutral hand and high NOG confidence required."
                )
    finally:
        if worker is not None:
            worker.request_stop()
            worker.thread.join(timeout=2)
            if not worker.done.is_set() or worker.thread.is_alive():
                print(
                    "Camera/native release NOT confirmed; no further local actions are authorized.",
                    file=sys.stderr,
                )
                status = 3
        cv.destroyAllWindows()
    return status


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.camera < 0:
        cli.error("camera index must be nonnegative")
    if not args.verify_only and not args.allow_local_demo:
        cli.error(
            "launch requires --allow-local-demo; use --verify-only for a read-only check"
        )
    try:
        from backend.workflows.deployment import handoff_assets, verify_assets
        from backend.workflows.workspace import workspace

        assets = handoff_assets(args.run)
        verify_assets(**assets, shadow=True)
        if args.verify_only:
            print("交接校验完成；未打开相机、界面或授予动作权限。")
            return 0
        perception = args.perception_model or workspace().path(
            workspace().settings.app.perception_model
        )
        if not perception.is_file():
            raise ValueError("missing local MediaPipe model; pass --perception-model")
        print(
            "实验性内置翻页：C开相机 → E启用 → O选中立手 → 预热/释放。空格停止控制；Esc关闭相机退出。"
        )
        print("未连接PowerPoint/WPS，不注入系统按键；真实模型性能与现场安全尚未验收。")
        return run_ui(
            assets, perception_model=perception, camera=args.camera, slides=args.slides
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            f"Local demo refused/failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
