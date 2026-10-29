"""Explicit source clocks. Never repair duplicate timestamps by inventing time."""
from time import monotonic_ns
from uuid import uuid4


class StreamClock:
    def __init__(self, now=monotonic_ns):
        self.now = now
        self.reset()

    def reset(self):
        self.stream_id = str(uuid4())
        self.frame_id = -1
        self.last_ns = None

    def stamp(self, timestamp_ns=None):
        value = self.now() if timestamp_ns is None else timestamp_ns
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("source timestamp must be nonnegative int64 ns")
        if self.last_ns is not None and value <= self.last_ns:
            raise ValueError("duplicate or backwards source timestamp")
        self.frame_id += 1
        self.last_ns = value
        return self.stream_id, self.frame_id, value
