"""Real application-local slide state, not keyboard injection or a mock command log."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from threading import Event

from pydantic import Field

from shared.config import load_config
from shared.contracts import Action, ActionResult, Capability
from shared.presentation_protocol import Deck


class PageActionResult(ActionResult):
    page_before: int = Field(ge=1, le=32)
    page_after: int = Field(ge=1, le=32)
    revision: int = Field(ge=0)


@dataclass(frozen=True)
class LocalPlatform:
    allowed_targets: tuple[str, ...]
    capabilities: tuple[Capability, ...] = (Capability.PRESENTATION,)
    scope: str = "application-local"


@dataclass(frozen=True)
class LocalDispatchConfig:
    actions: object
    platform: LocalPlatform


def demo_deck():
    data = files("shared").joinpath("defaults/presentation_demo.json").read_bytes()
    return Deck.model_validate_json(data), sha256(data).hexdigest()


class PresentationBackend:
    """State mutations are bounded and use the same gate as dispatcher commit.

    Stop cannot undo an action already committed. There are no held keys, OS
    handles or external callbacks here; a stopped/old-context call is rejected.
    """

    def __init__(self, controller, deck, deck_id, *, clock):
        self.controller, self.deck, self.deck_id, self.clock = (
            controller,
            deck,
            deck_id,
            clock,
        )
        self.target = "local-presentation:" + deck_id
        self.page = 1
        self.revision = 0
        self.armed_epoch = None
        self.stop_requested = Event()
        self.stop_requested.set()

    def config(self):
        return LocalDispatchConfig(load_config().actions, LocalPlatform((self.target,)))

    def capabilities(self):
        return (Capability.PRESENTATION,)

    def arm(self, context, *, owner_track_id=None):
        with self.controller.gate:
            if (
                context.target_context != self.target
                or not context.enabled
                or context.owner_track_id != owner_track_id
            ):
                raise ValueError("explicit local presentation owner/context required")
            self.armed_epoch = context.context_epoch
            self.stop_requested.clear()

    def target_available(self, target):
        return target == self.target and not self.stop_requested.is_set()

    def execute(self, command, now_ns):
        with self.controller.gate:
            context = self.controller.snapshot()
            now = (
                self.clock()
            )  # do not reuse a timestamp captured before waiting for the gate
            reason = None
            if self.stop_requested.is_set() or not context.enabled or context.suspended:
                reason = "local_presentation_stopped"
            elif (command.stream_id, command.context_epoch, command.timebase) != (
                context.stream_id,
                self.armed_epoch,
                context.timebase,
            ):
                reason = "local_presentation_stale_context"
            elif (
                context.context_epoch != self.armed_epoch
                or command.target_context != self.target
            ):
                reason = "local_presentation_wrong_target"
            elif (
                command.action not in {Action.NEXT_SLIDE, Action.PREVIOUS_SLIDE}
                or command.arguments
            ):
                reason = "local_presentation_unsupported_action"
            elif not command.issued_at_ns <= now < command.expires_at_ns:
                reason = "local_presentation_expired"
            before = self.page
            if reason is not None:
                return PageActionResult(
                    command_id=command.command_id,
                    accepted=False,
                    injection_status="not_attempted",
                    error_code=reason,
                    completed_at_ns=now,
                    acknowledgement_level="none",
                    timebase=command.timebase,
                    page_before=before,
                    page_after=before,
                    revision=self.revision,
                )
            delta = 1 if command.action == Action.NEXT_SLIDE else -1
            self.page = min(len(self.deck.slides), max(1, self.page + delta))
            self.revision += (
                1  # even a boundary command needs a distinct visual receipt
            )
            return PageActionResult(
                command_id=command.command_id,
                accepted=True,
                injection_status="confirmed",
                completed_at_ns=now,
                acknowledgement_level="application",
                timebase=command.timebase,
                page_before=before,
                page_after=self.page,
                revision=self.revision,
            )

    def release_all(self):
        self.stop_requested.set()
        with self.controller.gate:
            self.armed_epoch = None
        return True
