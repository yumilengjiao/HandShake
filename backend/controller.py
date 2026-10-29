"""Minimal lifecycle/context owner. T17 adds the actual interaction state machine."""

from shared.contracts import ContextChange, RuntimeContext


class ApplicationController:
    def __init__(self, stream_id: str, *, timebase="monotonic", target_context="mock-presentation"):
        self._context = RuntimeContext(stream_id=stream_id, timebase=timebase, context_epoch=0, target_context=target_context)
        self.changes: list[ContextChange] = []

    def snapshot(self) -> RuntimeContext:
        """Capture BEFORE submitting work; consumers must never re-stamp its epoch."""
        return self._context.model_copy(deep=True)

    def _change(self, reason: str, timestamp_ns: int, **updates) -> RuntimeContext:
        values = self._context.model_dump()
        values.update(updates, context_epoch=self._context.context_epoch + 1)
        new_context = RuntimeContext.model_validate(values)
        change = ContextChange(
            stream_id=new_context.stream_id, context_epoch=new_context.context_epoch,
            timebase=new_context.timebase, timestamp_ns=timestamp_ns, reason=reason,
        )
        self._context = new_context
        self.changes.append(change)
        return self.snapshot()

    def enable(self, timestamp_ns: int) -> RuntimeContext:
        return self._change("enable", timestamp_ns, enabled=True, owner_track_id=None, suspended=False)

    def request_mock_owner(self, track_id: str, timestamp_ns: int) -> RuntimeContext:
        if not self._context.enabled or self._context.suspended:
            raise ValueError("cannot bind owner while disabled/suspended")
        return self._change("owner", timestamp_ns, owner_track_id=track_id)

    def change_mode(self, mode: str, timestamp_ns: int) -> RuntimeContext:
        return self._change("mode", timestamp_ns, mode=mode, owner_track_id=None)

    def change_model(self, timestamp_ns: int) -> RuntimeContext:
        return self._change("model", timestamp_ns, owner_track_id=None)

    def suspend(self, timestamp_ns: int) -> RuntimeContext:
        return self._change("suspend", timestamp_ns, suspended=True, owner_track_id=None)

    def stop(self, timestamp_ns: int) -> RuntimeContext:
        return self._change("stop", timestamp_ns, enabled=False, owner_track_id=None)

    def reconnect(self, stream_id: str, timestamp_ns: int) -> RuntimeContext:
        if stream_id == self._context.stream_id:
            raise ValueError("reconnect requires a new stream_id")
        return self._change("reconnect", timestamp_ns, stream_id=stream_id, enabled=False, owner_track_id=None)
