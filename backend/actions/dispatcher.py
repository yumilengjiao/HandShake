"""Single bounded queue/worker with stop-priority serial commit and late cleanup.

The commit gate decides order, not cancellation of an already committed external
call. Independent release has its own bounded worker; a timed-out backend keeps
this dispatcher disabled until an explicit re-enable after safe quiescence.
"""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, Thread, Lock
import time

from backend.actions.audit import AuditTrail
from backend.actions.backends.dry_run import DryRunBackend
from shared.config import load_config
from shared.contracts import Action, ActionResult


@dataclass(frozen=True)
class Ticket:
    command_id: str
    future: Future


class Dispatcher:
    def __init__(self, controller, *, backend=None, config=None, clock=None, capacity=32, timeout_ns=500_000_000, focus_check=None, audit_sink=None):
        if type(capacity) is not int or capacity < 1 or type(timeout_ns) is not int or timeout_ns <= 0:
            raise ValueError("queue capacity and timeout must be positive integers")
        self.controller, self.config = controller, config or load_config()
        self.backend = backend or DryRunBackend()
        self.clock = clock or time.monotonic_ns
        self.focus_check = focus_check or (lambda target: target.startswith("mock-"))
        self.capacity, self.timeout_ns = capacity, timeout_ns
        self.gate = controller.gate
        self._stop = Event()
        self._closed = False
        self._queue = deque()
        self._wake = Event()
        self._release_wake = Event()
        self._release_lock = Lock()
        self._release_generation = 0
        self._released_generation = 0
        self._release_ok = True
        self._inflight = None
        self._seen_commands, self._seen_events, self._sequences = set(), set(), {}
        self._possible_held = set()
        self.results, self.safety = deque(maxlen=2048), AuditTrail(sink=audit_sink)
        self._worker = Thread(target=self._run, daemon=True, name="handshake-dispatch")
        self._releaser = Thread(target=self._run_release, daemon=True, name="handshake-release")
        self._worker.start()
        self._releaser.start()

    @property
    def stopped(self):
        return self._stop.is_set()

    def _result(self, command, reason, *, accepted=False, uncertain=False):
        return ActionResult(command_id=command.command_id, accepted=accepted,
                            injection_status="uncertain" if uncertain else "not_attempted",
                            error_code=reason, completed_at_ns=self.clock(), acknowledgement_level="none", timebase=command.timebase)

    def _validate(self, command):
        context = self.controller.snapshot()
        now = self.clock()
        if self._closed or self._stop.is_set():
            return "stopped"
        if not context.enabled or context.suspended or context.owner_track_id is None:
            return "inactive_context"
        if (command.stream_id, command.context_epoch, command.timebase) != (context.stream_id, context.context_epoch, context.timebase):
            return "stale_context"
        if not command.issued_at_ns <= now < command.expires_at_ns:
            return "expired_or_future_command"
        if command.target_context != context.target_context or context.target_context not in self.config.platform.allowed_targets:
            return "target_not_allowed"
        if command.action not in self.config.actions.allowlist or command.action not in set(self.config.actions.mappings.get(context.mode, {}).values()):
            return "action_or_mode_not_allowed"
        try:
            if not set(command.required_capabilities).issubset(set(self.backend.capabilities()) & set(self.config.platform.capabilities)):
                return "BACKEND_UNAVAILABLE"
            if not self.focus_check(command.target_context):
                return "target_focus_unavailable"
        except Exception:
            return "backend_or_focus_probe_failed"
        if len(self._seen_commands) >= 65536:
            return "dedup_capacity_requires_new_dispatcher"
        if command.command_id in self._seen_commands:
            return "duplicate_command"
        if command.control_stream_id is None:
            if command.source_event_id in self._seen_events:
                return "duplicate_event"
        else:
            if command.action not in {Action.MOVE_POINTER, Action.SET_VOLUME, Action.SCROLL}:
                return "invalid_continuous_action"
            previous = self._sequences.get(command.control_stream_id, -1)
            if command.update_seq <= previous:
                return "stale_continuous_update"
        latest = self.controller.snapshot()
        if self._stop.is_set():
            return "stopped"
        if latest != context:
            return "stale_context"
        return None

    def submit(self, command):
        future = Future()
        with self.gate:
            reason = self._validate(command)
            if reason is None and len(self._queue) >= self.capacity:
                reason = "queue_full"
            if reason:
                future.set_result(self._result(command, reason))
            else:
                self._queue.append((command, future))
                self._wake.set()
        return Ticket(command.command_id, future)

    def _cancel_queue(self, reason):
        while self._queue:
            command, future = self._queue.popleft()
            if not future.done():
                future.set_result(self._result(command, reason))

    def _request_release(self):
        with self._release_lock:
            self._release_generation += 1
            self._release_ok = False
        self._release_wake.set()

    def stop(self, now_ns=None, *, suspend=False):
        # Stop priority is observable even while a submitter holds the commit gate.
        self._stop.set()
        received = self.clock() if now_ns is None else now_ns
        with self.gate:
            context = self.controller.snapshot()
            if suspend:
                if not context.suspended:
                    self.controller.suspend(received)
            elif context.enabled or context.owner_track_id is not None or context.suspended:
                self.controller.stop(received)
            self._cancel_queue("stopped_before_commit")
            self.safety.append(dict(stop_received_ns=received, submissions_disabled_ns=self.clock(),
                                    inflight=self._inflight is not None, reason="suspend" if suspend else "stop"))
        self._request_release()

    def enable(self, now_ns=None):
        with self.gate:
            with self._release_lock:
                safe = self._release_ok and self._release_generation == self._released_generation
            if self._closed or self._inflight is not None or not safe or self._possible_held:
                raise RuntimeError("backend has not confirmed safe quiescence")
            self.controller.enable(self.clock() if now_ns is None else now_ns)
            self._stop.clear()

    def tick(self):
        with self.gate:
            expired = not self._stop.is_set() and self._inflight is not None and self.clock() - self._inflight[2] > self.timeout_ns
            if expired:
                command, future, _ = self._inflight
                if not future.done():
                    future.set_result(self._result(command, "backend_timeout_call_may_complete_late", accepted=True, uncertain=True))
        if expired:
            self.stop()

    def _run(self):
        while True:
            self._wake.wait()
            with self.gate:
                if self._closed and not self._queue:
                    return
                if not self._queue:
                    self._wake.clear()
                    continue
                command, future = self._queue.popleft()
                reason = self._validate(command)
                if reason:
                    future.set_result(self._result(command, reason))
                    continue
                # This is the documented irrevocable commit boundary. Stop after
                # this point cannot undo the call; late completion is released.
                self._seen_commands.add(command.command_id)
                if command.control_stream_id is None:
                    self._seen_events.add(command.source_event_id)
                else:
                    self._sequences[command.control_stream_id] = command.update_seq
                if command.action in {Action.BUTTON_DOWN, Action.CLICK}:
                    self._possible_held.add(command.command_id)
                self._inflight = (command, future, self.clock())
            uncertain = False
            try:
                result = self.backend.execute(command, self.clock())
                if not isinstance(result, ActionResult) or result.command_id != command.command_id or result.timebase != command.timebase:
                    raise ValueError("backend result identity mismatch")
                if result.completed_at_ns < command.issued_at_ns:
                    raise ValueError("backend completion precedes command")
                uncertain = result.injection_status.value in {"uncertain", "failed"}
            except Exception:
                uncertain = True
                result = self._result(command, "backend_exception_side_effect_unknown", accepted=True, uncertain=True)
            with self.gate:
                self.results.append(result)
                if uncertain:
                    self.stop()
                elif self._stop.is_set() or command.action in {Action.CLICK, Action.BUTTON_UP}:
                    # Publish cleanup before marking the call quiescent; close
                    # must keep the release worker alive for late side effects.
                    self._request_release()
                self._inflight = None
                if not future.done():
                    future.set_result(result)

    def _run_release(self):
        while True:
            self._release_wake.wait()
            self._release_wake.clear()
            with self._release_lock:
                generation = self._release_generation
            try:
                confirmed = self.backend.release_all() is True
                error = None if confirmed else "release_not_confirmed"
            except Exception:
                confirmed, error = False, "release_failed"
            with self.gate:
                if confirmed:
                    self._possible_held.clear()
                else:
                    self._stop.set()
                    context = self.controller.snapshot()
                    if context.enabled or context.owner_track_id is not None:
                        self.controller.stop(self.clock())
                    self._cancel_queue("release_failed")
                self.safety.append(dict(release_finished_ns=self.clock(), confirmed=confirmed, error=error, generation=generation,
                                        safe_quiescent=confirmed and self._inflight is None and not self._possible_held))
                with self._release_lock:
                    self._released_generation = generation
                    self._release_ok = confirmed
                    pending = self._release_generation > generation
                if self._closed and self._inflight is None and not pending:
                    return
            if pending:
                self._release_wake.set()

    def safety_status(self):
        with self.gate:
            with self._release_lock:
                released = self._release_ok and self._release_generation == self._released_generation
            return dict(submissions_disabled=self._stop.is_set(), inflight=self._inflight is not None,
                        release_confirmed=released, safe_quiescent=released and self._inflight is None
                        and not self._possible_held and not self._queue)

    def wait_released(self, timeout=1.0):
        # Test/CLI shutdown helper; never used from a stop or UI callback.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._release_lock:
                if self._release_generation == self._released_generation:
                    return self._release_ok
            time.sleep(0.001)
        return False

    def close(self):
        self.stop()
        with self.gate:
            self._closed = True
        self._wake.set()
        self._release_wake.set()
        self.safety.close()
