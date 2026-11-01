"""Manual application-local presentation service with one T18 dispatcher."""

import time
import uuid
from collections import OrderedDict, deque

from backend.actions.backends.presentation import (
    PageActionResult,
    PresentationBackend,
    demo_deck,
)
from backend.actions.dispatcher import Dispatcher
from backend.controller import ApplicationController
from shared.contracts import Action, ActionCommand, Capability
from shared.presentation_protocol import PageResult, PresentationSnapshot

SNAPSHOT_TTL_NS = 500_000_000
HEARTBEAT_NS = 1_500_000_000
RENDER_TIMEOUT_NS = 1_000_000_000


class PresentationSession:
    def __init__(self, *, clock=time.monotonic_ns, deck=None, deck_id=None):
        self.clock = clock
        self.session_id = uuid.uuid4().hex
        if deck is None:
            deck, deck_id = demo_deck()
        self.controller = ApplicationController(
            self.session_id, target_context="local-presentation:" + deck_id
        )
        self.controller.changes = deque(maxlen=128)
        self.backend = PresentationBackend(self.controller, deck, deck_id, clock=clock)
        self.manual_grant = False
        self.dispatcher = Dispatcher(
            self.controller,
            backend=self.backend,
            config=self.backend.config(),
            clock=clock,
            capacity=1,
            focus_check=self.backend.target_available,
            authorization_check=self._authorized,
        )
        self.last_command_id = self.sequence = 0
        self.last_heartbeat = clock()
        self.rendered = None
        self.render_since = clock()
        self.pending = None
        self.last_action = None
        self.reason = "initial"
        self.closed = False
        self.tokens = OrderedDict()

    def _authorized(self, context):
        return (
            self.manual_grant
            and context.owner_track_id is None
            and context.mode == "presentation"
            and context.target_context == self.backend.target
            and not self.backend.stop_requested.is_set()
            and self.rendered == (context.context_epoch, self.backend.revision)
            and self.clock() - self.last_heartbeat < HEARTBEAT_NS
        )

    def _stop(self, reason):
        self.backend.stop_requested.set()  # observable before waiting for commit gate
        self.manual_grant = False
        self.dispatcher.stop(self.clock())
        self.rendered = None
        self.reason = reason

    def fail(self, reason):
        self._stop(reason)
        self.closed = True

    def close(self):
        self._stop("shutdown")
        self.closed = True
        self.dispatcher.close()

    def _poll(self):
        self.dispatcher.tick()
        if self.pending is not None:
            request, ticket, action, before = self.pending
            if ticket.future.done():
                result = ticket.future.result()
                self.pending = None
                after = (
                    result.page_after
                    if isinstance(result, PageActionResult)
                    else self.backend.page
                )
                revision = (
                    result.revision
                    if isinstance(result, PageActionResult)
                    else self.backend.revision
                )
                if result.injection_status.value in {"uncertain", "failed"}:
                    outcome = "uncertain"
                elif not result.accepted:
                    outcome = "rejected"
                else:
                    outcome = (
                        "application_updated"
                        if before != after
                        else "boundary_no_change"
                    )
                self.last_action = PageResult(
                    request_id=str(request),
                    action=action.value,
                    outcome=outcome,
                    reason=result.error_code,
                    page_before=before,
                    page_after=after,
                    revision=str(revision),
                )
        now = self.clock()
        context = self.controller.snapshot()
        if context.enabled:
            if self.dispatcher.stopped:
                self._stop("dispatcher_stopped")
            elif now - self.last_heartbeat >= HEARTBEAT_NS:
                self._stop("heartbeat_timeout")
            elif (
                self.rendered != (context.context_epoch, self.backend.revision)
                and now - self.render_since >= RENDER_TIMEOUT_NS
            ):
                self._stop("render_timeout")
        if now - self.last_heartbeat >= 5_000_000_000:
            self.fail("parent_timeout")

    def accept(self, command):
        # Stop/shutdown do not wait for a frame/render receipt or a model.
        if command.session_id != self.session_id:
            raise ValueError("wrong presentation session")
        if command.kind in {"stop", "shutdown"}:
            self.backend.stop_requested.set()
        with self.controller.gate:
            if self.closed:
                return
            request = int(command.command_id)
            if request <= self.last_command_id:
                raise ValueError("replayed presentation command")
            self.last_command_id = request
            self._poll()
            if self.closed:
                return
            if command.kind in {"stop", "shutdown"}:
                self._stop("user_stop" if command.kind == "stop" else "shutdown")
                self.closed = command.kind == "shutdown"
                return
            context = self.controller.snapshot()
            token = self.tokens.get(int(command.based_on_sequence))
            if token is None or self.clock() - token[0] >= SNAPSHOT_TTL_NS:
                self._stop("stale_snapshot")
                return
            if (int(command.epoch), int(command.revision)) != (
                context.context_epoch,
                self.backend.revision,
            ) or token[1:] != (context.context_epoch, self.backend.revision):
                self.reason = "stale_context"
                return
            self.last_heartbeat = self.clock()
            if command.kind == "heartbeat":
                return
            if command.kind == "render_ack":
                self.rendered = (context.context_epoch, self.backend.revision)
                return
            if command.kind == "play":
                if context.enabled:
                    self.reason = "already_playing"
                    return
                if self.rendered != (context.context_epoch, self.backend.revision):
                    self.reason = "surface_not_ready"
                    return
                try:
                    self.dispatcher.enable(self.clock())
                except RuntimeError:
                    self.reason = "release_pending"
                    return
                self.backend.arm(self.controller.snapshot())
                self.manual_grant = True
                self.rendered = None
                self.render_since = self.clock()
                self.reason = "manual_playing"
                return
            if (
                not context.enabled
                or not self.manual_grant
                or self.backend.stop_requested.is_set()
            ):
                self.reason = "not_playing"
                return
            if self.pending is not None or self.rendered != (
                context.context_epoch,
                self.backend.revision,
            ):
                self.reason = "awaiting_render_or_result"
                return
            if not self._authorized(context):
                self._stop("manual_authorization_unavailable")
                return
            action = (
                Action.NEXT_SLIDE if command.kind == "next" else Action.PREVIOUS_SLIDE
            )
            now = self.clock()
            expiry = min(now + SNAPSHOT_TTL_NS, token[0] + SNAPSHOT_TTL_NS)
            if expiry <= now:
                self._stop("stale_snapshot")
                return
            value = ActionCommand(
                command_id=f"{self.session_id}:{request}",
                source_event_id=f"manual:{request}",
                stream_id=context.stream_id,
                context_epoch=context.context_epoch,
                action=action,
                issued_at_ns=now,
                expires_at_ns=expiry,
                required_capabilities=[Capability.PRESENTATION],
                target_context=context.target_context,
            )
            ticket = self.dispatcher.submit(value)
            self.pending = (request, ticket, action, self.backend.page)
            self.render_since = now
            self.reason = "command_submitted"

    def snapshot(self):
        with self.controller.gate:
            self._poll()
            context = self.controller.snapshot()
            now = self.clock()
            self.sequence += 1
            self.tokens[self.sequence] = (
                now,
                context.context_epoch,
                self.backend.revision,
            )
            while len(self.tokens) > 32:
                self.tokens.popitem(last=False)
            safety = self.dispatcher.safety_status()
            return PresentationSnapshot(
                session_id=self.session_id,
                sequence=str(self.sequence),
                epoch=str(context.context_epoch),
                last_command_id=str(self.last_command_id),
                timestamp_ns=str(now),
                state="playing" if context.enabled else "stopped",
                reason=self.reason,
                deck_id=self.backend.deck_id,
                deck_title=self.backend.deck.title,
                page=self.backend.page,
                page_count=len(self.backend.deck.slides),
                revision=str(self.backend.revision),
                rendered_revision=str(self.rendered[1])
                if self.rendered is not None
                and self.rendered[0] == context.context_epoch
                else None,
                busy=self.pending is not None,
                release_confirmed=safety["safe_quiescent"],
                slide=self.backend.deck.slides[self.backend.page - 1],
                last_action=self.last_action,
            )
