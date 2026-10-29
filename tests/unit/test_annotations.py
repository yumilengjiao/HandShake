import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.annotations import AnnotationInterval, AnnotationTimeline
from shared.contracts import Gesture


@pytest.fixture
def timeline():
    path = Path(__file__).resolve().parents[1] / "fixtures/annotation_examples.json"
    return AnnotationTimeline.model_validate_json(path.read_text())


def test_examples_have_all_classes_and_independent_intent(timeline):
    assert AnnotationTimeline.model_validate_json(timeline.model_dump_json()) == timeline
    assert {r.gesture_label for r in timeline.intervals if r.class_valid} == set(Gesture)
    left = [r for r in timeline.intervals if r.gesture_label == Gesture.SWIPE_LEFT]
    assert {r.intent_label for r in left} == {0, 1}
    assert all(r.target_command is None for r in left if r.intent_label == 0)
    wake = next(r for r in timeline.intervals if r.target_command == "request_control")
    assert wake.reference_event_time_ns - wake.start_ns == 600_000_000
    short_wake = next(r for r in timeline.intervals if "hold_too_short_for_event" in r.quality_tags)
    assert short_wake.class_valid and short_wake.reference_event_time_ns is None


@pytest.mark.parametrize("change", [
    {"intent_label": 0}, {"intent_label": True}, {"intent_label": 1.0},
    {"intent_label": "1"}, {"intent_label": "ambiguous"},
    {"reference_event_time_ns": 1_800_000_000}, {"end_ns": 1_000_000_000},
    {"target_command": "next_slide"}, {"class_valid": False},
    {"phase": "release"}, {"target_command": None},
    {"timebase": "monotonic"}, {"annotation_version": "v2"},
])
def test_bad_labels_or_event_times_fail(timeline, change):
    row = next(r for r in timeline.intervals if r.target_command == "request_control").model_dump(mode="json")
    with pytest.raises(ValidationError):
        AnnotationInterval(**(row | change))


def test_same_track_overlaps_fail_but_adjacent_intervals_work(timeline):
    data = timeline.model_dump(mode="json")
    data["intervals"][1]["start_ns"] = 999_999_999
    with pytest.raises(ValidationError, match="overlap"):
        AnnotationTimeline(**data)


def test_different_annotators_may_label_same_time(timeline):
    data = timeline.model_dump(mode="json")
    other = data["intervals"][0] | {"annotator_id": "second-annotator"}
    AnnotationTimeline(**(data | {"intervals": data["intervals"] + [other]}))


def test_one_recording_cannot_change_subject(timeline):
    data = timeline.model_dump(mode="json")
    data["intervals"][-1]["subject_id"] = "different-person"
    with pytest.raises(ValidationError, match="subject/session"):
        AnnotationTimeline(**data)
