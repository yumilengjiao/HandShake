"""Acquisition/perception collection runner, isolated from all action dispatch.

The processing thread and camera thread are independent of UI polling. Finite
shutdown deadlines expose native-call hangs; incomplete recordings are retained.
"""
from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
import random
from threading import Event, Lock, Thread
from time import monotonic_ns

from backend.capture.worker import CaptureWorker
from backend.collection.recording_writer import RecordingWriter, AsyncRecorder
from backend.perception.pipeline import PerceptionPipeline
from shared.contracts import Gesture, PerceptionStatus


class CollectionSession:
    def __init__(self, source, detector, *, recording=None, seed=42, max_frame_age_ns=500_000_000):
        if max_frame_age_ns <= 0:
            raise ValueError("frame age budget must be positive")
        self.source, self.detector = source, detector
        self.capture = CaptureWorker(source, backpressure=getattr(source, "video", None) is not None)
        self.pipeline = PerceptionPipeline(detector)
        self.recording_options = recording
        self.max_age_ns = max_frame_age_ns
        self.seed = seed
        self.prompts = [g.value for g in Gesture if g != Gesture.NO_ACTION]
        random.Random(seed).shuffle(self.prompts)
        self.prompt_index = -1
        self.paused = False
        self.prompt = "background / no prompt"
        self.stop_event, self.done = Event(), Event()
        self.lock = Lock()
        self.latest = None
        self.recorder = None
        self.recording_path = None
        self.error = None
        self.cancelled = False
        self.complete = False
        self.processed = 0
        self.consumed = 0
        self.age_samples_ns = deque(maxlen=4096)
        self.status_counts = {s.value:0 for s in PerceptionStatus}
        self.origin_ns = 0
        self.current_source_ns = 0
        self.last_progress_ns = monotonic_ns()
        self.started_ns = self.last_progress_ns
        self.thread = None

    def start(self):
        if self.thread:
            raise RuntimeError("session is single-use")
        # Permission checks occur BEFORE camera open and before files are created.
        if self.recording_options and not self.recording_options.get("consent_keypoints"):
            raise PermissionError("collection needs explicit keypoint consent")
        self.thread = Thread(target=self._run, name="handshake-perception", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.origin_ns = monotonic_ns() if getattr(self.source, "video", None) is None and not self.source.metadata.get("synthetic") else 0
            self.current_source_ns = self.origin_ns
            self.capture.start()
            if self.stop_event.is_set():
                return
            if self.recording_options:
                options = dict(self.recording_options)
                profile = {"source": self.source.metadata, "perception": self.detector.metadata,
                           "quality": asdict(self.pipeline.quality), "tracking": asdict(self.pipeline.association.settings),
                           "max_frame_age_ns": self.max_age_ns, "prompt_seed": self.seed,
                           "base_config_sha256": options.pop("config_sha256", None)}
                digest = sha256(json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()
                writer = RecordingWriter(**options, origin_ns=self.origin_ns,
                    source_timebase="synthetic" if self.source.metadata.get("synthetic") else "recording_relative" if getattr(self.source, "video", None) else "monotonic",
                    source=self.source.metadata, perception=self.detector.metadata | {"quality": profile["quality"], "tracking": profile["tracking"]},
                    synthetic=bool(self.detector.metadata.get("synthetic") or self.source.metadata.get("synthetic")), config_sha256=digest)
                self.recorder = AsyncRecorder(writer)
                self.recording_path = writer.root
                self.mark("start", prompt_seed=self.seed, prompt_order=self.prompts, context_available=False)
            while not self.stop_event.is_set():
                skipped, frame = self.capture.take(.1)
                for row in skipped:
                    self._record(row)
                if self.capture.stats.status_overflow:
                    raise RuntimeError("capture status ledger overflow; incomplete timeline")
                if frame is None:
                    if self.capture.done:
                        break
                    if monotonic_ns() - self.last_progress_ns > 3_000_000_000:
                        raise TimeoutError("no camera progress for 3 seconds")
                    continue
                self.current_source_ns = frame.info.timestamp_ns
                self.last_progress_ns = monotonic_ns()
                self.consumed += 1
                age_ns = max(0, monotonic_ns() - frame.received_at_ns)
                if self.source.metadata.get("kind") == "camera":
                    self.age_samples_ns.append(age_ns)
                if self.source.metadata.get("kind") == "camera" and age_ns > self.max_age_ns:
                    row = self.pipeline.skipped(frame.info, "stale_before_perception")
                else:
                    row = self.pipeline.process(frame)
                    if row.status.perception_status != PerceptionStatus.NOT_PROCESSED:
                        self.processed += 1
                self.last_progress_ns = monotonic_ns()
                if self.stop_event.is_set():
                    row = self.pipeline.skipped(frame.info, "stopped_during_perception")
                self._record(row, frame.rgb)
                with self.lock:
                    self.latest = (frame, row)
                if row.status.perception_status == PerceptionStatus.ERROR:
                    raise RuntimeError(row.status.drop_or_error_reason)
            if self.capture.error:
                raise RuntimeError(self.capture.error)
        except Exception as error:
            self.error = self.error or f"{type(error).__name__}: {error}"
        finally:
            if not self.capture.stop():
                self.error = self.error or self.capture.error
            try:
                skipped, frame = self.capture.take(0)
                for row in skipped:
                    self._record(row)
                if frame is not None:
                    self._record(self.pipeline.skipped(frame.info, "stopped_before_perception"), frame.rgb)
                if self.capture.stats.status_overflow:
                    self.error = self.error or "capture status ledger overflow"
                if self.recorder:
                    self.mark("cancel" if self.cancelled else "end", error=self.error)
            except Exception as error:
                self.error = self.error or str(error)
            try:
                self.pipeline.close()
            except Exception as error:
                self.error = self.error or f"perception close failed: {error}"
            if self.recorder:
                try:
                    stats = self.report()
                    stats.pop("complete", None)  # final completion is authoritative only in the manifest
                    self.complete = self.recorder.close(reason=self.error, cancelled=self.cancelled, stats=stats)
                    self.error = self.error or self.recorder.error
                except Exception as error:
                    self.error = self.error or str(error)
            self.done.set()

    def _record(self, row, rgb=None):
        self.current_source_ns = max(self.current_source_ns, row.info.timestamp_ns)
        self.status_counts[row.status.perception_status.value] += 1
        if self.recorder:
            raw = rgb if self.recorder.writer.manifest.consent_raw else None
            self.recorder.submit("append", row, raw_rgb=raw)

    def snapshot(self):
        with self.lock:
            return self.latest

    def mark(self, kind, **payload):
        if self.recorder:
            timestamp = monotonic_ns() if self.recorder.writer.manifest.source_timebase == "monotonic" else self.current_source_ns
            self.recorder.submit("event", kind, timestamp, **payload)

    def next_prompt(self, *, practice=False):
        if self.paused:
            return
        self.prompt_index += 1
        self.prompt = self.prompts[self.prompt_index % len(self.prompts)]
        self.mark("prompt", target_gesture=self.prompt, practice=practice, prompt_index=self.prompt_index)

    def toggle_pause(self):
        self.paused = not self.paused
        # Pause affects prompts, not capture: background/time remain recorded.
        self.mark("pause" if self.paused else "resume", capture_continues=True)

    def request_stop(self, *, reason=None, cancelled=False):
        self.error = self.error or reason
        self.cancelled = self.cancelled or cancelled
        self.stop_event.set()

    def stop(self, timeout=5.):
        self.request_stop()
        if self.thread:
            self.thread.join(timeout)
            if self.thread.is_alive():
                self.error = self.error or "perception/native cleanup deadline exceeded; not safely cancelled"
                return False
        return True

    def report(self):
        elapsed = max(1, monotonic_ns() - self.started_ns)
        ages = sorted(self.age_samples_ns)
        age_report = {"clock": "local_read_complete_to_consumer_not_exposure", "last_n_samples": len(ages),
                      "sample_capacity": 4096, "p50_ns": ages[len(ages)//2] if ages else None,
                      "p95_ns": ages[min(len(ages)-1, int(len(ages)*.95))] if ages else None,
                      "max_ns": ages[-1] if ages else None}
        return {"capture": self.capture.stats.as_dict(), "processed_frames": self.processed,
                "consumed_frames": self.consumed, "consumer_hz": self.consumed * 1e9 / elapsed,
                "frame_age": age_report, "status_counts": dict(self.status_counts),
                "elapsed_ns": elapsed, "error": self.error, "complete": self.complete,
                "source": dict(self.source.metadata), "perception": dict(self.detector.metadata),
                "synthetic": bool(self.detector.metadata.get("synthetic") or self.source.metadata.get("synthetic")),
                "control_backend": "none (collection cannot inject actions)"}
