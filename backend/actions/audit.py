"""Bounded diagnostic ring; optional I/O never runs on commit/stop/release paths."""
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Thread


class AuditTrail(deque):
    def __init__(self, *, sink=None, maxlen=2048, capacity=128):
        super().__init__(maxlen=maxlen)
        self.sink = sink
        self.failures = 0
        self.dropped = 0
        self._queue = Queue(maxsize=capacity)
        self._closed = Event()
        if sink is not None:
            Thread(target=self._run, name="handshake-audit", daemon=True).start()

    def append(self, record):
        super().append(record)
        if self.sink is not None:
            if self._closed.is_set():
                self.dropped += 1
                return
            try:
                self._queue.put_nowait(dict(record))
            except Full:
                self.dropped += 1

    def _run(self):
        while not self._closed.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=.05)
            except Empty:
                continue
            try:
                self.sink(record)
            except Exception:
                self.failures += 1
            finally:
                self._queue.task_done()

    def close(self):
        # Optional external sink may be stuck; no join on safety/shutdown path.
        self._closed.set()
