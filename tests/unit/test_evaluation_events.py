"""Independent event truth, hand counts and brute-force matching oracle. No devices."""

import itertools
import math
import random

import pytest

from algorithm.evaluation.event_contracts import (
    EventInput,
    PredictedEvent,
    RecordingPlan,
    TargetEvent,
)
from algorithm.evaluation.events import (
    evaluate_events,
    intersect,
    match_events,
    subtract,
    union,
)


def identity(event_id, **changes):
    return (
        dict(
            event_id=event_id,
            recording_id="r",
            timebase="synthetic",
            target_context="deck",
            mode="presentation",
            kind="command",
            action="next_slide",
        )
        | changes
    )


def target(event_id="t", stamp=100, **changes):
    return TargetEvent(
        **(identity(event_id) | dict(start_ns=stamp, confirmation_ns=stamp) | changes)
    )


def prediction(event_id="p", stamp=100, **changes):
    return PredictedEvent(
        **(
            identity(event_id)
            | dict(
                scope="dry_run",
                submitted_ns=stamp,
                external_injection_status="not_attempted",
            )
            | changes
        )
    )


def recording(**changes):
    return RecordingPlan(
        **(
            dict(
                recording_id="r",
                timebase="synthetic",
                start_ns=0,
                end_ns=1000,
                valid_intervals=((0, 1000),),
                non_operation_intervals=((0, 1000),),
                enabled_intervals=((0, 1000),),
                active_intervals=(),
                unavailable_intervals=(),
                denominator_reference="independent-protocol",
                validity_origin="protocol_or_recording_damage",
            )
            | changes
        )
    )


def inputs(targets=(), predictions=(), **changes):
    return EventInput(
        **(
            dict(
                format_version="event-evaluation-v1",
                synthetic=True,
                protocol_reference="hand-calculated-fixture",
                early_ns=-10,
                late_ns=20,
                recordings=(recording(),),
                targets=tuple(targets),
                predictions=tuple(predictions),
            )
            | changes
        )
    )


def layer(data, name="submitted", kind="command"):
    return evaluate_events(data)["kinds"][kind]["layers"][name]


def test_perfect_all_missed_all_wrong_duplicates_and_empty():
    exact = layer(inputs([target()], [prediction()]))
    assert (exact["tp"], exact["fp"], exact["fn"], exact["f1"]) == (1, 0, 0, 1)
    missed = layer(inputs([target()]))
    assert (missed["tp"], missed["fp"], missed["fn"], missed["recall"]) == (0, 0, 1, 0)
    wrong = layer(inputs([target()], [prediction(action="previous_slide")]))
    assert (wrong["tp"], wrong["fp"], wrong["fn"]) == (0, 1, 1)
    repeated = layer(inputs([target()], [prediction("p1"), prediction("p2", 101)]))
    assert (repeated["tp"], repeated["fp"], repeated["fn"]) == (1, 1, 0)
    assert len(repeated["duplicate_predictions"]) == 1
    assert (
        repeated["background_false_events"] == 0
    )  # duplicate still FP, not protocol background
    empty = layer(inputs())
    assert (
        empty["f1"] is None and empty["precision"] is None and empty["recall"] is None
    )


@pytest.mark.parametrize(
    "stamp,matched", [(89, 0), (90, 1), (100, 1), (120, 1), (121, 0)]
)
def test_inclusive_matching_time_boundaries(stamp, matched):
    assert layer(inputs([target()], [prediction(stamp=stamp)]))["tp"] == matched


def test_non_greedy_matching_maximizes_cardinality_then_time_cost():
    # t1 can use p1 or p2; t2 can only use p1. Nearest-first for t1 loses t2.
    data = inputs(
        [target("t1", 100), target("t2", 110)],
        [prediction("p1", 101), prediction("p2", 90)],
        early_ns=-10,
        late_ns=1,
    )
    result = layer(data)
    assert result["tp"] == 2
    assert [(p["target_id"], p["prediction_id"]) for p in result["matched"]] == [
        ("t1", "p2"),
        ("t2", "p1"),
    ]


def test_matching_agrees_with_exhaustive_small_graph_oracle_and_is_order_invariant():
    rng = random.Random(15)
    for _ in range(70):
        ts = [target(f"t{i}", rng.randrange(10, 25)) for i in range(rng.randrange(5))]
        ps = [
            prediction(f"p{i}", rng.randrange(5, 30)) for i in range(rng.randrange(5))
        ]
        best = (0, 0)
        for size in range(1, min(len(ts), len(ps)) + 1):
            for selected in itertools.combinations(range(len(ts)), size):
                for assigned in itertools.permutations(range(len(ps)), size):
                    deltas = [
                        ps[j].submitted_ns - ts[i].confirmation_ns
                        for i, j in zip(selected, assigned)
                    ]
                    if all(-5 <= d <= 8 for d in deltas):
                        best = min(best, (-size, sum(abs(d) for d in deltas)))
        a, b, pairs, _ = match_events(
            ts, ps, timestamp_field="submitted_ns", early_ns=-5, late_ns=8
        )
        assert (
            -len(pairs),
            sum(abs(b[j].submitted_ns - a[i].confirmation_ns) for i, j in pairs),
        ) == best
        other = match_events(
            list(reversed(ts)),
            list(reversed(ps)),
            timestamp_field="submitted_ns",
            early_ns=-5,
            late_ns=8,
        )
        assert pairs == other[2]


@pytest.mark.parametrize(
    "changes",
    [
        {"target_context": "other"},
        {"mode": "music"},
        {"recording_id": "other"},
        {"timebase": "monotonic"},
        {"kind": "control", "action": "stop"},
    ],
)
def test_context_recording_clock_mode_action_and_kind_do_not_cross_match(changes):
    pred = prediction(**changes)
    records = (recording(),)
    if "recording_id" in changes or "timebase" in changes:
        records += (
            recording(
                **{
                    k: v
                    for k, v in changes.items()
                    if k in {"recording_id", "timebase"}
                }
            ),
        )
    result = layer(inputs([target()], [pred], recordings=records))
    assert result["tp"] == 0 and result["fn"] == 1


def test_dry_run_local_and_external_execution_evidence_are_separate():
    dry = prediction("dry", 100, accepted_ns=101)
    local = prediction(
        "local", 100, scope="application-local", accepted_ns=101, application_ns=105
    )
    external = prediction(
        "ext",
        100,
        scope="external",
        accepted_ns=101,
        injected_ns=102,
        application_ns=105,
        external_injection_status="confirmed",
        source_ns=90,
        prediction_ns=95,
    )
    for pred, expected in [
        (dry, [1, 0, 0, 0, 0]),
        (local, [1, 1, 0, 1, 0]),
        (external, [1, 1, 1, 0, 1]),
    ]:
        report = evaluate_events(inputs([target()], [pred]))["kinds"]["command"][
            "layers"
        ]
        assert [
            report[name]["tp"]
            for name in (
                "submitted",
                "accepted",
                "external_injected",
                "application_local",
                "external_application",
            )
        ] == expected
    injected = layer(inputs([target(start_ns=80)], [external]), "external_injected")
    assert injected["latency"]["source_to_injection"]["p50_ns"] == 12
    assert injected["latency"]["reference_to_injection"]["p50_ns"] == 2
    assert injected["latency"]["action_start_to_injection"]["p50_ns"] == 22
    app = layer(inputs([target()], [external]), "external_application")
    assert app["latency"]["injection_to_application"]["p95_ns"] == 3
    assert (
        layer(inputs([target()], [dry]), "external_injected")["latency"][
            "source_to_injection"
        ]["count"]
        == 0
    )


def test_injection_matching_uses_injection_not_submission_time():
    pred = prediction(
        scope="external",
        accepted_ns=101,
        injected_ns=130,
        external_injection_status="confirmed",
    )
    data = inputs([target()], [pred])
    assert layer(data)["tp"] == 1
    assert layer(data, "external_injected")["tp"] == 0
    assert layer(data, "external_injected")["fn"] == 1


@pytest.mark.parametrize("status", ["failed", "uncertain"])
def test_failed_uncertain_execution_cannot_count_as_injection(status):
    data = inputs(
        [target()],
        [
            prediction(
                scope="external", accepted_ns=101, external_injection_status=status
            )
        ],
    )
    assert layer(data, "accepted")["tp"] == 1
    assert layer(data, "external_injected")["tp"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"injected_ns": 101},
        {"scope": "dry_run", "application_ns": 101},
        {
            "scope": "application-local",
            "injected_ns": 101,
            "external_injection_status": "confirmed",
        },
        {"scope": "external", "application_ns": 105, "accepted_ns": 101},
        {"prediction_ns": 200},
        {"external_injection_status": "confirmed"},
    ],
)
def test_invalid_stage_evidence_rejected(changes):
    with pytest.raises(ValueError):
        prediction(**changes)


def test_background_denominator_union_damage_exclusion_and_no_active_filter():
    rec = recording(
        valid_intervals=((0, 700), (600, 1000)),
        non_operation_intervals=((0, 500), (400, 1000)),
        active_intervals=((0, 5),),
        unavailable_intervals=((500, 1000),),
    )
    data = inputs(
        [target(start_ns=80)],
        [prediction("good"), prediction("false", 500)],
        recordings=(rec,),
    )
    result = layer(data)
    assert result["background_valid_ns"] == 959  # exclude [80,121), once
    assert result["background_false_events"] == 1
    assert result["background_false_per_hour"] == pytest.approx(3_600_000_000_000 / 959)
    changed = inputs(
        data.targets,
        data.predictions,
        recordings=(rec.model_copy(update={"active_intervals": ()}),),
    )
    assert layer(changed)["background_valid_ns"] == result["background_valid_ns"]
    assert evaluate_events(data)["availability"][0]["unavailable_ns"] == 500


def test_twenty_minutes_zero_false_events_does_not_prove_one_per_hour():
    end = 20 * 60 * 10**9
    rec = recording(
        end_ns=end,
        valid_intervals=((0, end),),
        non_operation_intervals=((0, end),),
        enabled_intervals=((0, end),),
    )
    result = layer(inputs(recordings=(rec,)))
    assert result["background_false_per_hour"] == 0
    assert result["zero_count_poisson_upper95_per_hour"] == pytest.approx(
        -math.log(0.05) * 3
    )
    assert result["zero_count_poisson_upper95_per_hour"] > 1
    zero = layer(inputs(recordings=(recording(non_operation_intervals=()),)))
    assert (
        zero["background_false_per_hour"] is None
        and zero["zero_count_poisson_upper95_per_hour"] is None
    )


def test_invalid_domains_damage_and_conflicting_availability():
    with pytest.raises(ValueError, match="duplicate"):
        inputs([target(), target()])
    with pytest.raises(ValueError, match="synthetic"):
        inputs(synthetic=False)
    with pytest.raises(ValueError, match="outside"):
        inputs(
            [target(start_ns=1)],
            recordings=(
                recording(
                    start_ns=50,
                    valid_intervals=((50, 1000),),
                    non_operation_intervals=(),
                    enabled_intervals=(),
                ),
            ),
        )
    with pytest.raises(ValueError, match="valid recording"):
        layer(
            inputs([target()], recordings=(recording(valid_intervals=((500, 1000),)),))
        )
    with pytest.raises(ValueError, match="conflict"):
        layer(
            inputs(
                recordings=(
                    recording(
                        active_intervals=((0, 50),), unavailable_intervals=((20, 80),)
                    ),
                )
            )
        )
    invalid = layer(
        inputs(
            predictions=[prediction()],
            recordings=(recording(valid_intervals=((500, 1000),)),),
        )
    )
    assert invalid["outside_valid_predictions"] == 1 and invalid["fp"] == 0


def test_interval_arithmetic():
    assert union([(0, 5), (3, 9), (9, 10)]) == [(0, 10)]
    assert intersect([(0, 5), (7, 10)], [(3, 8)]) == [(3, 5), (7, 8)]
    assert subtract([(0, 10)], [(3, 4), (2, 3), (6, 20)]) == [(0, 2), (4, 6)]
