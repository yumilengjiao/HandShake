"""Offline-only T05 annotations. Never imported by the online decision chain."""

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from shared import PROTOCOL_VERSION
from shared.contracts import Action, Contract, Gesture, Identifier, Ns


class AnnotationInterval(Contract):
    timebase: Literal["recording_relative"] = "recording_relative"
    annotation_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    recording_id: Identifier
    subject_id: Identifier
    session_id: Identifier
    hand_track_reference: Identifier
    annotator_id: Identifier
    start_ns: Ns
    end_ns: Ns
    gesture_label: Gesture | None
    class_valid: StrictBool
    intent_label: Literal[0, 1, "ambiguous"]
    intent_valid: StrictBool
    phase: Literal["background", "preparation", "active", "release", "ambiguous"]
    phase_valid: StrictBool
    reference_event_time_ns: Ns | None = None
    target_command: Action | Literal["request_control", "exit_control"] | None = None
    quality_tags: list[Identifier] = Field(default_factory=list)
    environment_tags: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_bool_intent(cls, data):
        if isinstance(data, dict):
            intent = data.get("intent_label")
            if not (type(intent) is int and intent in (0, 1)) and intent != "ambiguous":
                raise ValueError("intent_label requires integer 0/1 or ambiguous")
        return data

    @model_validator(mode="after")
    def check_annotation(self):
        if self.start_ns >= self.end_ns:
            raise ValueError("annotation requires [start_ns,end_ns) with start < end")
        if self.class_valid != (self.gesture_label is not None):
            raise ValueError("unknown class must be null with class_valid=false")
        if self.intent_valid != (self.intent_label != "ambiguous"):
            raise ValueError("ambiguous intent must be masked; known intent must be valid")
        if self.phase_valid != (self.phase != "ambiguous"):
            raise ValueError("ambiguous phase must be masked")
        if (self.reference_event_time_ns is None) != (self.target_command is None):
            raise ValueError("reference event and target command must appear together")
        if self.reference_event_time_ns is not None:
            if not self.start_ns <= self.reference_event_time_ns < self.end_ns:
                raise ValueError("reference event must fall in [start,end)")
            if self.intent_label != 1 or not self.class_valid or self.phase != "active":
                raise ValueError("reference event requires an active valid intentional gesture")
            permitted = {
                Gesture.WAKE_OPEN: {"request_control"},
                Gesture.EXIT_POSE: {"exit_control"},
                Gesture.SWIPE_LEFT: {Action.PREVIOUS_SLIDE, Action.PREVIOUS_TRACK},
                Gesture.SWIPE_RIGHT: {Action.NEXT_SLIDE, Action.NEXT_TRACK},
            }
            if self.target_command not in permitted.get(self.gesture_label, set()):
                raise ValueError("target command does not match the v1 gesture protocol")
        return self


class AnnotationTimeline(Contract):
    timebase: Literal["recording_relative"] = "recording_relative"
    intervals: list[AnnotationInterval]

    @model_validator(mode="after")
    def check_overlaps(self):
        grouped = {}
        identities = {}
        for interval in self.intervals:
            identity = (interval.subject_id, interval.session_id)
            if identities.setdefault(interval.recording_id, identity) != identity:
                raise ValueError("one recording cannot change subject/session")
            key = (interval.recording_id, interval.hand_track_reference, interval.annotator_id)
            grouped.setdefault(key, []).append(interval)
        for group in grouped.values():
            ordered = sorted(group, key=lambda item: item.start_ns)
            if any(a.end_ns > b.start_ns for a, b in zip(ordered, ordered[1:])):
                raise ValueError("same recording/track/annotator intervals overlap")
        return self
