"""Chunked recording writer and bounded asynchronous disk adapter."""
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from time import monotonic
from uuid import uuid4
import os

import numpy as np

from shared import PROTOCOL_VERSION
from shared.recordings import (ChunkEntry, CollectionEvent, RecordingManifest, atomic_json, digest_file,
                               load_chunk, relative_frame, save_chunk, save_npz)


class RecordingWriter:
    def __init__(self, root, *, subject_id, session_id, consent_reference, consent_keypoints=False,
                 consent_raw=False, synthetic=False, origin_ns=0, source_timebase="monotonic",
                 source=None, perception=None, code_version="unknown", config_sha256="0"*64,
                 chunk_frames=32, chunk_duration_ns=2_000_000_000):
        if not consent_keypoints:
            raise PermissionError("explicit keypoint consent is required before creating a recording")
        if chunk_frames < 1 or chunk_duration_ns < 1:
            raise ValueError("positive chunk bounds required")
        recording_id = str(uuid4())
        self.manifest = RecordingManifest(
            recording_id=recording_id, subject_id=subject_id, session_id=session_id,
            consent_reference=consent_reference, consent_keypoints=True, consent_raw=consent_raw,
            synthetic=synthetic, created_utc=datetime.now(timezone.utc).isoformat(),
            protocol_version=PROTOCOL_VERSION, code_version=code_version, config_sha256=config_sha256,
            origin_ns=origin_ns, source_timebase=source_timebase, source=source or {}, perception=perception or {})
        self.root = Path(root) / recording_id
        self.root.mkdir(parents=True, mode=0o700, exist_ok=False)
        (self.root / "chunks").mkdir()
        if consent_raw:
            (self.root / "raw").mkdir()
        self.chunk_frames, self.chunk_duration_ns = chunk_frames, chunk_duration_ns
        self.pending, self.raw = [], {}
        self.last_info = None
        self.event_seq = 0
        self.closed = False
        self._publish()
        self.events = (self.root / "events.jsonl").open("x", encoding="utf-8")

    def _publish(self, **updates):
        self.manifest = RecordingManifest.model_validate(self.manifest.model_dump() | updates)
        atomic_json(self.root / "manifest.json", self.manifest.model_dump(mode="json"))

    def append(self, frame, *, raw_rgb=None):
        if self.closed:
            raise RuntimeError("recording is closed")
        if frame.info.timebase != self.manifest.source_timebase:
            raise ValueError("recording source timebase changed")
        if self.last_info and (frame.info.stream_id != self.last_info.stream_id or
                               frame.info.frame_id <= self.last_info.frame_id or
                               frame.info.timestamp_ns <= self.last_info.timestamp_ns):
            raise ValueError("recording requires one strictly ordered stream")
        if raw_rgb is not None:
            if not self.manifest.consent_raw:
                raise PermissionError("raw RGB needs separate consent")
            if raw_rgb.dtype != np.uint8 or raw_rgb.shape != (frame.info.height, frame.info.width, 3):
                raise ValueError("invalid raw RGB frame")
        row = relative_frame(frame, self.manifest.origin_ns)
        if raw_rgb is not None:
            self.raw[f"frame_{len(self.pending):06}"] = np.array(raw_rgb, copy=True)
        self.pending.append(row)
        self.last_info = frame.info
        if len(self.pending) >= self.chunk_frames or row.info.timestamp_ns - self.pending[0].info.timestamp_ns >= self.chunk_duration_ns:
            self.flush()

    def event(self, kind, timestamp_ns, **payload):
        if self.closed or type(timestamp_ns) is not int or timestamp_ns < self.manifest.origin_ns:
            raise ValueError("invalid recording event/time")
        if kind not in {"start", "pause", "resume", "end", "cancel", "prompt", "retry", "health", "stop"}:
            raise ValueError("unknown collection event")
        entry = CollectionEvent(sequence=self.event_seq, timestamp_ns=timestamp_ns - self.manifest.origin_ns,
                                kind=kind, payload=payload)
        self.events.write(entry.model_dump_json() + "\n")
        self.events.flush()
        self.event_seq += 1

    def flush(self):
        if not self.pending:
            return
        number = len(self.manifest.chunks)
        relative = f"chunks/{number:06}.npz"
        path = self.root / relative
        save_chunk(path, self.pending)
        reloaded = load_chunk(path)
        if len(reloaded) != len(self.pending):
            raise ValueError("chunk read-back validation failed")
        raw_path = raw_hash = None
        if self.raw:
            raw_path = f"raw/{number:06}.npz"
            save_npz(self.root / raw_path, self.raw)
            raw_hash = digest_file(self.root / raw_path)
        entry = ChunkEntry(path=relative, sha256=digest_file(path), frames=len(self.pending),
                           first_ns=self.pending[0].info.timestamp_ns, last_ns=self.pending[-1].info.timestamp_ns,
                           raw_path=raw_path, raw_sha256=raw_hash, raw_frames=len(self.raw))
        self.events.flush()
        os.fsync(self.events.fileno())
        self._publish(chunks=[c.model_dump(mode="json") for c in self.manifest.chunks] + [entry.model_dump(mode="json")],
                      frame_count=self.manifest.frame_count + len(self.pending))
        self.pending.clear()
        self.raw.clear()

    def finish(self, *, reason=None, cancelled=False, stats=None):
        if self.closed:
            return self.manifest.complete
        try:
            self.flush()
            self.events.flush()
            os.fsync(self.events.fileno())
            self.events.close()
            complete = reason is None and not cancelled and self.manifest.frame_count > 0
            self._publish(complete=complete, state="complete" if complete else "cancelled" if cancelled else "incomplete",
                          reason=reason or (None if complete else "cancelled_or_empty"),
                          events_sha256=digest_file(self.root / "events.jsonl"), stats=stats or {})
            return complete
        finally:
            self.closed = True
            self.events.close()


class AsyncRecorder:
    """One disk owner. Overflow stops collection, never silently loses records.

    Only the joining caller publishes complete after the disk worker has stopped;
    a timed-out writer cannot later turn an incomplete recording into a complete one.
    """
    def __init__(self, writer, capacity=128):
        if capacity < 1:
            raise ValueError("positive disk queue bound required")
        self.writer = writer
        self.queue = Queue(maxsize=capacity)
        self.error = None
        self.peak = 0
        self.dropped = 0
        self.closed = False
        self.timed_out = False
        self.thread = Thread(target=self._run, name="handshake-recording", daemon=True)
        self.thread.start()

    def submit(self, kind, *args, **kwargs):
        if self.closed or self.error:
            raise RuntimeError(self.error or "recorder is closed")
        if kind not in {"append", "event"}:
            raise ValueError("unsupported disk operation")
        try:
            self.queue.put_nowait((kind, args, kwargs))
            self.peak = max(self.peak, self.queue.qsize())
        except Full:
            self.dropped += 1
            self.error = "disk_queue_overflow: recording is incomplete"
            raise RuntimeError(self.error) from None

    def _run(self):
        try:
            while True:
                try:
                    item = self.queue.get(timeout=.1)
                except Empty:
                    if self.closed:
                        break
                    continue
                if item is None:
                    break
                kind, args, kwargs = item
                getattr(self.writer, kind)(*args, **kwargs)
            self.writer.flush()
        except Exception as error:
            self.error = f"disk failure: {type(error).__name__}: {error}"
            # Keep the last committed incomplete manifest; do not invent a success.
            self.writer.events.close()
        finally:
            if self.error or self.timed_out:
                self.writer.events.close()

    def close(self, *, reason=None, cancelled=False, stats=None, timeout=5.):
        if self.closed:
            return self.writer.manifest.complete
        self.closed = True
        deadline = monotonic() + timeout
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=max(0, deadline - monotonic()))
            except Full:
                self.error = self.error or "disk stop queue timed out"
            self.thread.join(max(0, deadline - monotonic()))
        if self.thread.is_alive():
            self.timed_out = True
            self.error = self.error or "disk flush timed out; incomplete retained"
            return False
        if self.error:
            self.writer.events.close()
            try:
                self.writer._publish(reason=self.error, stats=stats or {})
            except OSError:
                pass  # initial incomplete manifest still remains authoritative
            return False
        return self.writer.finish(reason=reason, cancelled=cancelled,
                                  stats=(stats or {}) | {"disk_queue_peak": self.peak, "disk_dropped": self.dropped})
