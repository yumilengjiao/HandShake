"""CLI adapters for T03/T04/T06. Imports no camera SDK until explicitly requested."""
import json
import math
import os
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

from backend.capture.camera import OpenCVSource, SyntheticSource
from backend.collection.session import CollectionSession
from backend.perception.mediapipe_adapter import MediaPipeDetector, SyntheticDetector
from shared.annotations import AnnotationTimeline
from shared.recordings import iter_recording, load_manifest, recover_recording


class NoPerception:
    metadata = {"kind": "disabled_no_model", "disabled": True, "synthetic": False}
    def close(self):
        pass


def add_arguments(parser):
    parser.add_argument("--source", choices=["camera", "video", "synthetic"], default="camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-clock", choices=["fps", "decoder"], default="fps")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--rotation", type=int, choices=[0,90,180,270], default=0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--model", type=Path, help="explicit local MediaPipe Hand Landmarker .task asset (not a gesture model)")
    parser.add_argument("--model-sha256", help="expected SHA256; mismatch refuses asset loading")
    parser.add_argument("--model-source", default="user-supplied-local-asset")
    parser.add_argument("--model-license", default="verify license before redistribution")
    parser.add_argument("--duration", type=float, default=30., help="wall-clock seconds, finite and positive")
    parser.add_argument("--frames", type=int, default=90, help="finite synthetic input frame count")
    parser.add_argument("--headless", action="store_true", help="do not open an OpenCV preview window")
    parser.add_argument("--report", type=Path, help="exclusive JSON probe/report output; no camera pixels")
    parser.add_argument("--output", type=Path, default=Path("data/recordings"))
    parser.add_argument("--subject", help="anonymous subject ID; no names")
    parser.add_argument("--session", help="anonymous session ID")
    parser.add_argument("--consent-keypoints", action="store_true", help="confirm authorized saving of hand keypoints")
    parser.add_argument("--consent-reference", help="local anonymous consent record reference, NOT signed form contents")
    parser.add_argument("--save-raw", action="store_true", help="optional lossless RGB NPZ chunks; contains private camera images")
    parser.add_argument("--consent-raw", action="store_true", help="separate explicit permission for raw RGB storage")
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--recording", type=Path, help="recording directory for inspect/recover/annotate")
    parser.add_argument("--allow-incomplete", action="store_true", help="explicit inspection only; never marks data complete")
    parser.add_argument("--labels", type=Path, help="manual AnnotationTimeline JSON to validate/export as a new revision")
    parser.add_argument("--edit-labels", type=Path, help="load an existing annotation revision into the skeleton editor")
    parser.add_argument("--annotator", default="annotator-local")
    parser.add_argument("--hand-reference", default="manual-hand-1")


def run_data_command(args):
    if args.recording is None:
        raise ValueError("--recording is required")
    if args.mode == "inspect":
        manifest = load_manifest(args.recording, allow_incomplete=args.allow_incomplete)
        count = sum(1 for _ in iter_recording(args.recording, allow_incomplete=args.allow_incomplete))
        print(json.dumps({"recording_id": manifest.recording_id, "complete": manifest.complete,
                          "verified_frames": count, "chunks": len(manifest.chunks), "synthetic": manifest.synthetic}))
    elif args.mode == "recover":
        print(recover_recording(args.recording))
    else:
        from backend.collection.annotation import annotation_view, export_annotations
        if args.labels:
            timeline = AnnotationTimeline.model_validate_json(args.labels.read_text())
            print(export_annotations(args.recording, timeline))
        elif args.headless:
            raise ValueError("headless annotation requires --labels; inspect does not edit labels")
        else:
            annotation_view(args.recording, annotator_id=args.annotator, hand_reference=args.hand_reference, existing=args.edit_labels)
    return 0


def run_capture_command(args, config, *, code_version):
    if args.duration <= 0 or not math.isfinite(args.duration):
        raise ValueError("--duration must be finite and positive")
    if args.report and args.report.exists():
        raise FileExistsError("report already exists; choose a new path")
    recording = None
    if args.mode == "collect":
        if not args.consent_keypoints or not all((args.subject, args.session, args.consent_reference)):
            raise ValueError("unavailable: collect requires --consent-keypoints, --consent-reference, --subject and --session")
        if args.save_raw and not args.consent_raw:
            raise PermissionError("--save-raw requires separate --consent-raw authorization")
        if args.source != "synthetic" and args.model is None:
            raise ValueError("collect requires a local --model; missing assets never fall back to synthetic hands")
        if config.privacy.save_raw_video and not (args.save_raw and args.consent_raw):
            raise PermissionError("raw configuration never substitutes for explicit raw consent")
        recording = dict(root=args.output, subject_id=args.subject, session_id=args.session,
                         consent_reference=args.consent_reference, consent_keypoints=True,
                         consent_raw=args.save_raw and args.consent_raw, code_version=code_version,
                         config_sha256=config.digest())
    elif args.save_raw or args.consent_keypoints or args.consent_raw:
        raise ValueError("preview/probe never records pixels or landmarks; use explicit collect")
    if args.source == "synthetic":
        source = SyntheticSource(frames=args.frames, fps=args.fps, width=args.width, height=args.height, paced=True)
    else:
        if args.source == "video" and args.video is None:
            raise ValueError("--source video requires --video")
        source = OpenCVSource(args.camera, video=args.video if args.source == "video" else None,
                              width=args.width, height=args.height, fps=args.fps, rotation=args.rotation,
                              preview_mirrored=not args.no_mirror, video_clock=args.video_clock)
    detector = (MediaPipeDetector(args.model, expected_sha256=args.model_sha256, source=args.model_source,
                                 license_note=args.model_license, max_hands=config.runtime.max_hands)
                if args.model else SyntheticDetector() if args.source == "synthetic" else NoPerception())
    session = CollectionSession(source, detector, recording=recording, seed=args.prompt_seed)
    window = None
    try:
        if not args.headless:
            if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and __import__("sys").platform != "darwin":
                raise RuntimeError("no display; use --headless")
            from backend.collection.views import PreviewWindow
            window = PreviewWindow()
            print("q/ESC: finish | x: cancel | p: pause prompts (capture continues) | n: next prompt | r: practice/retry", flush=True)
        session.start()
        started = monotonic()
        while not session.done.is_set():
            if window:
                window.show(session)
            if monotonic() - started >= args.duration:
                session.request_stop()
            if monotonic_ns() - session.last_progress_ns > 5_000_000_000:
                session.request_stop(reason="processing watchdog expired (native call may remain blocked)")
            if session.stop_event.is_set():
                break
            sleep(.01)
    except KeyboardInterrupt:
        session.request_stop(cancelled=True)
    finally:
        stopped = session.stop()
        if session.thread is None:
            detector.close()
        if window:
            window.close()
    report = session.report()
    report["shutdown_confirmed"] = stopped
    if session.recording_path:
        report["recording_path"] = str(session.recording_path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive file, so concurrent invocations cannot overwrite evidence.
        with args.report.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, allow_nan=False, indent=2)
            out.write("\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 2 if session.error or not stopped or (recording and not session.complete) else 0
