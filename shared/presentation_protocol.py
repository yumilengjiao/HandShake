"""Private presentation IPC v1: manual application-local control only."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared.desktop_protocol import Counter, SessionId, WireModel


class Slide(WireModel):
    title: str = Field(min_length=1, max_length=200)
    lines: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def bounded_text(self):
        if any(not line or len(line) > 300 for line in self.lines):
            raise ValueError("slide text budget exceeded")
        return self


class Deck(WireModel):
    format_version: Literal["presentation-deck-v1"]
    title: str = Field(min_length=1, max_length=200)
    slides: tuple[Slide, ...] = Field(min_length=1, max_length=32)


class PresentationCommand(WireModel):
    version: Literal[1]
    session_id: SessionId
    command_id: Counter
    epoch: Counter
    based_on_sequence: Counter
    revision: Counter
    kind: Literal[
        "play", "next", "previous", "stop", "heartbeat", "render_ack", "shutdown"
    ]


class PageResult(WireModel):
    request_id: Counter
    action: Literal["next_slide", "previous_slide"]
    outcome: Literal[
        "application_updated", "boundary_no_change", "rejected", "uncertain"
    ]
    reason: str | None = Field(default=None, max_length=128)
    page_before: int = Field(ge=1, le=32)
    page_after: int = Field(ge=1, le=32)
    revision: Counter


class PresentationSnapshot(WireModel):
    version: Literal[1] = 1
    session_id: SessionId
    sequence: Counter
    epoch: Counter
    last_command_id: Counter
    timestamp_ns: Counter
    scope: Literal["application-local"] = "application-local"
    control_origin: Literal["manual"] = "manual"
    external_control_available: Literal[False] = False
    gesture_control_available: Literal[False] = False
    state: Literal["stopped", "playing"]
    reason: str = Field(min_length=1, max_length=128)
    deck_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    deck_title: str = Field(min_length=1, max_length=200)
    page: int = Field(ge=1, le=32)
    page_count: int = Field(ge=1, le=32)
    revision: Counter
    rendered_revision: Counter | None
    busy: bool
    release_confirmed: bool
    slide: Slide
    last_action: PageResult | None

    @field_validator(
        "external_control_available", "gesture_control_available", mode="before"
    )
    @classmethod
    def no_external_authority(cls, value):
        if value is not False:
            raise ValueError("external/gesture control must be the boolean false")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.page > self.page_count:
            raise ValueError("page outside deck")
        if self.rendered_revision is not None and int(self.rendered_revision) > int(
            self.revision
        ):
            raise ValueError("render receipt from the future")
        if self.last_action is not None:
            result = self.last_action
            if max(result.page_before, result.page_after) > self.page_count or int(
                result.revision
            ) > int(self.revision):
                raise ValueError("action receipt outside current deck/revision")
            if (
                result.outcome == "application_updated"
                and result.page_before == result.page_after
            ) or (
                result.outcome == "boundary_no_change"
                and result.page_before != result.page_after
            ):
                raise ValueError("action outcome contradicts page transition")
        return self
