"""Latest-frame worker with a bounded, image-free dropped-frame ledger.

Only the worker touches the camera. Stop has a deadline: a blocked native driver
may outlive it. Such a worker is reported unhealthy, NEVER claimed safely cancelled.
"""
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import monotonic_ns

from shared.contracts import PerceptionStatus
from shared.observations import ObservationFrame


@dataclass
class CaptureStats:
    acquired: int = 0
    consumer_skipped: int = 0
    status_overflow: int = 0
    max_pending_frames: int = 0
    first_received_ns: int | None = None
    last_received_ns: int | None = None

    def as_dict(self):
        result = dict(vars(self))
        elapsed = (self.last_received_ns or 0) - (self.first_received_ns or 0)
        result["actual_capture_hz"] = (self.acquired - 1) * 1e9 / elapsed if elapsed > 0 else None
        result["driver_drops"] = None
        return result


class CaptureWorker:
    def __init__(self, source, *, status_capacity=512, backpressure=False):
        if status_capacity < 1:
            raise ValueError("status capacity must be positive")
        self.source, self.status_capacity = source, status_capacity
        self.backpressure = backpressure  # offline files: do not discard the whole video while decoding
        self.stats = CaptureStats()
        self._condition = Condition()
        self._stop = Event()
        self._ready = Event()
        self._latest = None
        self._skipped = deque()
        self._thread = None
        self.done = False
        self.error = None

    def start(self, timeout=5.):
        if self._thread is not None:
            raise RuntimeError("worker is single-use; reconnect with a new worker/source stream")
        self._thread = Thread(target=self._run, name="handshake-camera", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self._stop.set()
            raise TimeoutError("camera open timed out; native call may still be running")
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        try:
            self.source.open()
            self._ready.set()
            while not self._stop.is_set():
                if self.backpressure:
                    with self._condition:
                        self._condition.wait_for(lambda: self._latest is None or self._stop.is_set())
                    if self._stop.is_set():
                        break
                try:
                    frame = self.source.read()
                except StopIteration:
                    break
                with self._condition:
                    if self._latest is not None:
                        old = self._latest.info
                        status = old.status(PerceptionStatus.NOT_PROCESSED, reason="latest_frame_replaced")
                        if len(self._skipped) >= self.status_capacity:
                            self._skipped.popleft()
                            self.stats.status_overflow += 1
                        self._skipped.append(ObservationFrame(old, status))
                        self.stats.consumer_skipped += 1
                    self._latest = frame
                    self.stats.acquired += 1
                    if self.stats.first_received_ns is None:
                        self.stats.first_received_ns = frame.received_at_ns
                    self.stats.last_received_ns = frame.received_at_ns
                    self.stats.max_pending_frames = 1
                    self._condition.notify_all()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.source.close()
            except Exception as error:
                self.error = self.error or f"close failed: {error}"
            with self._condition:
                self.done = True
                self._ready.set()
                self._condition.notify_all()

    def take(self, timeout=.1):
        with self._condition:
            self._condition.wait_for(lambda: self._latest is not None or self.done, timeout)
            frame, self._latest = self._latest, None
            skipped = tuple(self._skipped)
            self._skipped.clear()
            self._condition.notify_all()
            return skipped, frame

    def stop(self, timeout=2.):
        self._stop.set()
        request_stop = getattr(self.source, "request_stop", None)
        if request_stop is not None:
            request_stop()  # cooperative wakeup only; never releases a native camera from another thread
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.error = self.error or "camera read/close did not stop before deadline"
                return False
        return True

    def idle_age_ns(self):
        last = self.stats.last_received_ns
        return None if last is None else monotonic_ns() - last
