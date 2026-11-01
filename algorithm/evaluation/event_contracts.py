"""Versioned event-evaluation inputs. Truth and time denominators come from protocol, not model state."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from shared.contracts import Action
from shared.desktop_protocol import WireModel

Ns = Annotated[int, Field(ge=0, le=2**63 - 1)]
Text = Annotated[str, Field(min_length=1, max_length=128)]
Interval = tuple[Ns, Ns]


class RecordingPlan(WireModel):
    recording_id: Text
    timebase: Literal["recording_relative", "monotonic", "synthetic"]
    start_ns: Ns
    end_ns: Ns
    valid_intervals: tuple[Interval, ...] = Field(max_length=1000)
    non_operation_intervals: tuple[Interval, ...] = Field(max_length=1000)
    enabled_intervals: tuple[Interval, ...] = Field(max_length=1000)
    active_intervals: tuple[Interval, ...] = Field(max_length=1000)
    unavailable_intervals: tuple[Interval, ...] = Field(max_length=1000)
    denominator_reference: Text
    validity_origin: Literal["protocol_or_recording_damage"]

    @model_validator(mode="after")
    def intervals_valid(self):
        if self.end_ns <= self.start_ns:
            raise ValueError("empty recording range")
        for name in (
            "valid_intervals",
            "non_operation_intervals",
            "enabled_intervals",
            "active_intervals",
            "unavailable_intervals",
        ):
            if any(
                not self.start_ns <= a < b <= self.end_ns
                for a, b in getattr(self, name)
            ):
                raise ValueError("interval outside recording")
        return self


class EventIdentity(WireModel):
    event_id: Text
    recording_id: Text
    timebase: Literal["recording_relative", "monotonic", "synthetic"]
    target_context: Text
    mode: Text
    kind: Literal["command", "control"]
    action: Text

    @model_validator(mode="after")
    def known_action(self):
        if self.kind == "command" and self.action not in {a.value for a in Action}:
            raise ValueError("unknown command action")
        if self.kind == "control" and self.action not in {
            "enable",
            "disable",
            "wake",
            "stop",
            "bind_owner",
            "play",
            "pause",
        }:
            raise ValueError("unknown control action")
        return self

    @property
    def match_group(self):
        return (
            self.recording_id,
            self.timebase,
            self.target_context,
            self.mode,
            self.kind,
            self.action,
        )


class TargetEvent(EventIdentity):
    start_ns: Ns
    confirmation_ns: Ns

    @model_validator(mode="after")
    def ordered(self):
        if self.start_ns > self.confirmation_ns:
            raise ValueError("target start after confirmation")
        return self


class PredictedEvent(EventIdentity):
    scope: Literal["dry_run", "application-local", "external"]
    source_ns: Ns | None = None
    prediction_ns: Ns | None = None
    submitted_ns: Ns
    accepted_ns: Ns | None = None
    external_injection_status: Literal[
        "not_attempted", "confirmed", "failed", "uncertain"
    ]
    injected_ns: Ns | None = None
    application_ns: Ns | None = None

    @model_validator(mode="after")
    def evidence_consistent(self):
        times = [
            getattr(self, field)
            for field in (
                "source_ns",
                "prediction_ns",
                "submitted_ns",
                "accepted_ns",
                "injected_ns",
                "application_ns",
            )
        ]
        known = [t for t in times if t is not None]
        if known != sorted(known):
            raise ValueError("event timestamps must be ordered on one timebase")
        if (self.external_injection_status == "confirmed") != (
            self.injected_ns is not None
        ):
            raise ValueError(
                "only confirmed external injection has an injection timestamp"
            )
        if self.injected_ns is not None and (
            self.scope != "external" or self.accepted_ns is None
        ):
            raise ValueError(
                "external injection needs actual external accepted execution"
            )
        if (
            self.scope != "external"
            and self.external_injection_status != "not_attempted"
        ):
            raise ValueError("local/dry-run cannot claim external injection")
        if self.application_ns is not None and (
            self.scope == "dry_run" or self.accepted_ns is None
        ):
            raise ValueError(
                "application response needs real accepted application activity"
            )
        if (
            self.application_ns is not None
            and self.scope == "external"
            and self.injected_ns is None
        ):
            raise ValueError("external response requires confirmed execution evidence")
        return self


class EventInput(WireModel):
    format_version: Literal["event-evaluation-v1"]
    synthetic: bool
    protocol_reference: Text
    early_ns: int = Field(ge=-10_000_000_000, le=0)
    late_ns: Ns = Field(le=10_000_000_000)
    recordings: tuple[RecordingPlan, ...] = Field(max_length=1000)
    targets: tuple[TargetEvent, ...] = Field(max_length=5000)
    predictions: tuple[PredictedEvent, ...] = Field(max_length=5000)

    @model_validator(mode="after")
    def identities(self):
        if not self.synthetic and any(
            r.timebase == "synthetic" for r in self.recordings
        ):
            raise ValueError("synthetic clock requires synthetic evaluation label")
        records = {(r.recording_id, r.timebase): r for r in self.recordings}
        if len(records) != len(self.recordings):
            raise ValueError("duplicate recording/timebase")
        for events in (self.targets, self.predictions):
            ids = [(e.recording_id, e.timebase, e.event_id) for e in events]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate event identity")
            for event in events:
                record = records.get((event.recording_id, event.timebase))
                if record is None:
                    raise ValueError("event without recording/timebase domain")
                if (
                    isinstance(event, TargetEvent)
                    and not record.start_ns <= event.start_ns < record.end_ns
                ):
                    raise ValueError("target start outside recording range")
                stamp = (
                    event.confirmation_ns
                    if isinstance(event, TargetEvent)
                    else event.submitted_ns
                )
                if not record.start_ns <= stamp < record.end_ns:
                    raise ValueError("event outside recording range")
        return self
