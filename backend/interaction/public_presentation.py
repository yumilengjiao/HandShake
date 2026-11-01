"""Synthetic-only host for public-gesture application-local presentation testing.

Not a camera/UI control entry point. No arbitrary backend injection, no external
capabilities. Live grants remain unavailable while semantics/event policy are draft.
"""

from collections import OrderedDict, deque
from threading import Event, Thread, current_thread
from uuid import uuid4

from backend.actions.backends.presentation import PresentationBackend, demo_deck
from backend.actions.dispatcher import Dispatcher
from backend.controller import ApplicationController
from backend.interaction.public_policy import PublicGestureEngine
from shared.contracts import Action, ActionCommand, Capability

SURFACE_TTL_NS = 500_000_000
HEARTBEAT_NS = 1_500_000_000
RENDER_TIMEOUT_NS = 1_000_000_000


class SyntheticPublicPresentation:
    def __init__(
        self,
        *,
        calibration,
        calibration_sha256,
        clock,
        allow_synthetic=False,
        policy=None,
        watchdog=True,
    ):
        self._validate_calibration_source(calibration, allow_synthetic)
        self._raw_clock = clock
        self._last_clock = 0
        self._revoked, self._closed = Event(), Event()
        self._revoked.set()
        self._read_clock()
        deck, deck_id = demo_deck()
        self.controller = ApplicationController(
            self._stream_id(),
            timebase=self._source_timebase(),
            target_context="local-presentation:" + deck_id,
        )
        self.controller.changes = deque(maxlen=128)
        # Validate policy/artifact before starting workers.
        self.engine = PublicGestureEngine(
            self.controller,
            calibration=calibration,
            calibration_sha256=calibration_sha256,
            policy=policy,
        )
        self.backend = PresentationBackend(
            self.controller, deck, deck_id, clock=self._dispatch_clock
        )
        self._grant_owner = None
        self.rendered = None
        self._render_since = self._last_heartbeat = self._last_clock
        self._sequence = 0
        self._tokens = OrderedDict()
        self.pending = None
        self.commands = deque(maxlen=128)
        self.last_result = None
        self.reason = "initial_synthetic_only"
        self.dispatcher = Dispatcher(
            self.controller,
            backend=self.backend,
            config=self.backend.config(),
            clock=self._dispatch_clock,
            capacity=1,
            focus_check=self.backend.target_available,
            authorization_check=self._authorized,
        )
        self.engine.stop_handler = self._suspend_dispatch
        self._watchdog = None
        self._workers_closed = False
        if watchdog:
            self._watchdog = Thread(
                target=self._watch, daemon=True, name="handshake-public-watchdog"
            )
            self._watchdog.start()

    def _validate_calibration_source(self, calibration, allow_synthetic):
        if (
            allow_synthetic is not True
            or not calibration.source.data_synthetic
            or not calibration.source.model_synthetic
        ):
            raise ValueError(
                "synthetic local execution requires explicit opt-in and synthetic calibration/model provenance"
            )

    def _stream_id(self):
        return "public-replay-" + uuid4().hex

    def _source_timebase(self):
        return "synthetic"

    def _arm_backend(self, context, track_id):
        self.backend.arm(context, owner_track_id=track_id)

    def _engine_tick(self, now_ns):
        self.engine.tick(now_ns)

    def _read_clock(self):
        try:
            now = self._raw_clock()
            if type(now) is not int or not self._last_clock <= now < 2**63:
                raise ValueError("synthetic host clock must be nondecreasing int64 ns")
        except Exception:
            self._revoked.set()
            if hasattr(self, "backend"):
                self.backend.stop_requested.set()
            raise
        self._last_clock = now
        return now

    def _dispatch_clock(self):
        try:
            return self._read_clock()
        except Exception:
            # Revoke BEFORE returning a fallback time for rejection/cleanup receipts.
            self._revoked.set()
            self.backend.stop_requested.set()
            return self._last_clock

    def _authorized(self, context):
        now = self._dispatch_clock()
        return (
            not self._closed.is_set()
            and not self._revoked.is_set()
            and self._grant_owner is not None
            and context.owner_track_id == self._grant_owner
            and context.timebase == "synthetic"
            and context.mode == "presentation"
            and context.target_context == self.backend.target
            and self.rendered == (context.context_epoch, self.backend.revision)
            and now - self._last_heartbeat < HEARTBEAT_NS
            and self.engine.last_good_ns is not None
            and 0 <= now - self.engine.last_good_ns <= self.engine.policy.max_age_ns
        )

    def _suspend_dispatch(self, now_ns, *, suspend=True):
        self._revoked.set()
        self.backend.stop_requested.set()
        self._grant_owner = None
        self.rendered = None
        self.dispatcher.stop(now_ns, suspend=suspend)

    def stop(self, reason="explicit_stop"):
        # These bits are visible even if the commit gate is currently occupied.
        self._revoked.set()
        self.backend.stop_requested.set()
        with self.controller.gate:
            if self._workers_closed:
                return
            self._suspend_dispatch(self._dispatch_clock(), suspend=False)
            self.engine.sync()
            self.reason = reason

    def _receipt(self, sequence, epoch, revision):
        if any(type(value) is not int for value in (sequence, epoch, revision)):
            return False
        token = self._tokens.get(sequence)
        context = self.controller.snapshot()
        now = self._read_clock()
        return (
            token is not None
            and 0 <= now - token[0] < SURFACE_TTL_NS
            and token[1:]
            == (epoch, revision)
            == (context.context_epoch, self.backend.revision)
        )

    def acknowledge_surface(self, *, sequence, epoch, revision):
        with self.controller.gate:
            if self._closed.is_set() or not self._receipt(sequence, epoch, revision):
                return False
            self.rendered = (epoch, revision)
            self._last_heartbeat = self._read_clock()
            return True

    def heartbeat(self, *, sequence, epoch, revision):
        with self.controller.gate:
            if self._closed.is_set() or not self._receipt(sequence, epoch, revision):
                return False
            self._last_heartbeat = self._read_clock()
            return True

    def enable(self):
        with self.controller.gate:
            now = self._read_clock()
            context = self.controller.snapshot()
            if self._closed.is_set() or context.enabled:
                return False
            if (
                self.rendered != (context.context_epoch, self.backend.revision)
                or now - self._last_heartbeat >= SURFACE_TTL_NS
            ):
                self.reason = "fresh_surface_receipt_required"
                return False
            self.dispatcher.enable(now)  # safe-quiescence barrier, no auto re-enable
            self.engine.sync()
            self._grant_owner = None
            self.rendered = None
            self._render_since = now
            self.reason = "waiting_explicit_owner"
            return True

    def arm_owner(self, track_id, prediction, frame, *, ready):
        with self.controller.gate:
            if self._closed.is_set():
                return False
            try:
                context = self.engine.arm_owner(
                    track_id, prediction, frame, ready=ready, now_ns=self._read_clock()
                )
                if context is None:
                    self.reason = self.engine.reason
                    return False
                self._arm_backend(context, track_id)
                self._grant_owner = track_id
                self._revoked.clear()
                self.rendered = None
                self._render_since = self._last_clock
                self.reason = "owner_bound_wait_surface_and_neutral"
                return True
            except Exception:
                self.stop("owner_input_rejected")
                raise

    def update(self, prediction, frame, *, ready):
        with self.controller.gate:
            if self._closed.is_set():
                return None, None
            try:
                self.tick()
                event = self.engine.update(
                    prediction, frame, ready=ready, now_ns=self._read_clock()
                )
                self.reason = self.engine.reason
                if event is None:
                    return None, None
                context = self.controller.snapshot()
                if self.pending is not None or not self._authorized(context):
                    self.reason = "event_discarded_no_surface_or_authority"
                    return event, None  # consumed, NEVER queue it for a future surface
                command = ActionCommand(
                    command_id="cmd:" + event.event_id,
                    source_event_id=event.event_id,
                    stream_id=event.stream_id,
                    context_epoch=event.context_epoch,
                    timebase="synthetic",
                    action=Action(event.action),
                    issued_at_ns=event.commit_ns,
                    expires_at_ns=min(
                        event.commit_ns + SURFACE_TTL_NS,
                        event.source_timestamp_ns + self.engine.policy.max_age_ns,
                    ),
                    required_capabilities=[Capability.PRESENTATION],
                    target_context=event.target_context,
                )
                self.commands.append(command)
                ticket = self.dispatcher.submit(command)
                self.pending = ticket
                self._render_since = self._last_clock
                self.reason = "synthetic_public_command_submitted"
                return event, ticket
            except Exception:
                self.stop("public_input_rejected")
                raise

    def tick(self):
        with self.controller.gate:
            try:
                now = self._read_clock()
                self.dispatcher.tick()
                if self.pending is not None and self.pending.future.done():
                    self.last_result = self.pending.future.result()
                    self.pending = None
                context = self.controller.snapshot()
                if context.enabled:
                    if (
                        self.dispatcher.stopped
                        or now - self._last_heartbeat >= HEARTBEAT_NS
                    ):
                        self.stop("dispatcher_or_heartbeat_timeout")
                    elif (
                        self.rendered != (context.context_epoch, self.backend.revision)
                        and now - self._render_since >= RENDER_TIMEOUT_NS
                    ):
                        self.stop("surface_response_timeout")
                    else:
                        self._engine_tick(now)
                        if self.controller.snapshot().suspended:
                            self.reason = self.engine.reason
            except Exception:
                self.stop("clock_or_watchdog_failure")
                raise

    def snapshot(self):
        with self.controller.gate:
            self.tick()
            context = self.controller.snapshot()
            self._sequence += 1
            self._tokens[self._sequence] = (
                self._last_clock,
                context.context_epoch,
                self.backend.revision,
            )
            while len(self._tokens) > 32:
                self._tokens.popitem(last=False)
            return {
                "format_version": "synthetic-public-presentation-v1",
                "synthetic": True,
                "scope": "application-local",
                "control_origin": "synthetic_public_policy",
                "live_available": False,
                "external_control_available": False,
                "state": self.engine.state,
                "reason": self.reason,
                "sequence": self._sequence,
                "epoch": context.context_epoch,
                "revision": self.backend.revision,
                "page": self.backend.page,
                "page_count": len(self.backend.deck.slides),
                "rendered": self.rendered
                == (context.context_epoch, self.backend.revision),
                "busy": self.pending is not None,
                "policy_sha256": self.engine.policy_sha256,
                "calibration_sha256": self.engine.calibration_sha256,
                "safety": self.dispatcher.safety_status(),
            }

    def _watch(self):
        while not self._closed.wait(0.02):
            try:
                self.tick()
            except Exception:
                self.close(reason="watchdog_failure")
                return

    def close(self, reason="shutdown"):
        self._closed.set()  # a dead watchdog can never authorize a restart
        self._revoked.set()
        self.backend.stop_requested.set()
        with self.controller.gate:
            if not self._workers_closed:
                self.stop(reason)
                self.dispatcher.close()
                self._workers_closed = True
        if self._watchdog is not None and self._watchdog is not current_thread():
            self._watchdog.join(timeout=0.2)
