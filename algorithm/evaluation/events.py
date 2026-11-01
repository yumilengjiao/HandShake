"""Deterministic maximum-cardinality/minimum-time-error event evaluation."""

import heapq
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

LAYERS = {
    "submitted": "submitted_ns",
    "accepted": "accepted_ns",
    "external_injected": "injected_ns",
    "application_local": "application_ns",
    "external_application": "application_ns",
}


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersect(left, right):
    a, b, i, j, result = union(left), union(right), 0, 0, []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(intervals, excluded):
    removed = union(excluded)
    output = []
    for start, end in union(intervals):
        cursor = start
        for a, b in removed:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                output.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            output.append((cursor, end))
    return output


def contains(intervals, stamp):
    return any(a <= stamp < b for a, b in intervals)


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def match_events(targets, predictions, *, timestamp_field, early_ns, late_ns):
    """Successive shortest augmenting paths; stable-ID node/edge ordering breaks ties.

    No greedy nearest-neighbour match: that can sacrifice a valid second pair.
    Budgets bound pathological dense bursts instead of silently approximating.
    """
    if (
        type(early_ns) is not int
        or type(late_ns) is not int
        or not -10_000_000_000 <= early_ns <= 0 <= late_ns <= 10_000_000_000
    ):
        raise ValueError("invalid event tolerance")
    truth = sorted(targets, key=lambda e: e.event_id)
    guessed = sorted(predictions, key=lambda e: e.event_id)
    if max(len(truth), len(guessed)) > 1000:
        raise ValueError("event match group exceeds 1000-event budget")
    n, m = len(truth), len(guessed)
    source, sink = n + m, n + m + 1
    graph = [[] for _ in range(n + m + 2)]

    def add(a, b, cost):
        forward = _Edge(b, len(graph[b]), 1, cost)
        graph[a].append(forward)
        graph[b].append(_Edge(a, len(graph[a]) - 1, 0, -cost))
        return forward

    for i in range(n):
        add(source, i, 0)
    for j in range(m):
        add(n + j, sink, 0)
    by_time = sorted(
        (getattr(event, timestamp_field), j) for j, event in enumerate(guessed)
    )
    times = [row[0] for row in by_time]
    eligible = {}
    for i, event in enumerate(truth):
        low, high = event.confirmation_ns + early_ns, event.confirmation_ns + late_ns
        js = sorted(
            j for _, j in by_time[bisect_left(times, low) : bisect_right(times, high)]
        )
        for j in js:
            if len(eligible) >= 25000:
                raise ValueError("eligible event edge budget exceeded")
            eligible[i, j] = add(
                i,
                n + j,
                abs(getattr(guessed[j], timestamp_field) - event.confirmation_ns),
            )
    potential = [0] * len(graph)
    while True:
        dist = [math.inf] * len(graph)
        prev = [None] * len(graph)
        dist[source] = 0
        queue = [(0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != dist[node]:
                continue
            for k, edge in enumerate(graph[node]):
                if not edge.capacity:
                    continue
                candidate = cost + edge.cost + potential[node] - potential[edge.to]
                if candidate < dist[edge.to]:
                    dist[edge.to] = candidate
                    prev[edge.to] = (node, k)
                    heapq.heappush(queue, (candidate, edge.to))
        if prev[sink] is None:
            break
        for node, cost in enumerate(dist):
            if cost != math.inf:
                potential[node] += cost
        node = sink
        while node != source:
            parent, k = prev[node]
            edge = graph[parent][k]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
    pairs = [(i, j) for (i, j), edge in eligible.items() if edge.capacity == 0]
    matched_truth, matched_pred = {i for i, _ in pairs}, {j for _, j in pairs}
    duplicates = {
        j for i, j in eligible if j not in matched_pred and i in matched_truth
    }
    return truth, guessed, pairs, duplicates


def _summary(values):
    return {
        "count": len(values),
        "p50_ns": float(np.percentile(values, 50)) if values else None,
        "p95_ns": float(np.percentile(values, 95)) if values else None,
        "min_ns": min(values) if values else None,
        "max_ns": max(values) if values else None,
    }


def evaluate_events(data):
    plans = {(r.recording_id, r.timebase): r for r in data.recordings}
    for event in data.targets:
        if not contains(
            union(plans[event.recording_id, event.timebase].valid_intervals),
            event.confirmation_ns,
        ):
            raise ValueError(
                "target confirmation is outside declared valid recording time"
            )
    for event in data.predictions:
        plan = plans[event.recording_id, event.timebase]
        for field in (*LAYERS.values(), "source_ns", "prediction_ns"):
            stamp = getattr(event, field)
            if stamp is not None and not plan.start_ns <= stamp < plan.end_ns:
                raise ValueError(
                    "observed event stage is outside its recording clock domain"
                )
    report = {
        "format_version": "event-report-v1",
        "synthetic": data.synthetic,
        "protocol_reference": data.protocol_reference,
        "early_ns": data.early_ns,
        "late_ns": data.late_ns,
        "matching": "max-cardinality then min absolute time error; stable-ID residual shortest paths",
        "time_policy": "same recording/timebase/action/mode/target/kind only; integer-ns inclusive matching bounds",
        "availability": [],
        "kinds": {},
        "execution_policy": "dry-run submitted only; application-local responses are not external execution",
    }
    for record in data.recordings:
        valid = union(record.valid_intervals)
        active = intersect(valid, record.active_intervals)
        unavailable = intersect(valid, record.unavailable_intervals)
        if duration(intersect(active, unavailable)):
            raise ValueError("ACTIVE and unavailable intervals conflict")
        if duration(subtract(active, record.enabled_intervals)):
            raise ValueError("ACTIVE intervals must be within protocol enabled time")
        report["availability"].append(
            {
                "recording_id": record.recording_id,
                "timebase": record.timebase,
                "valid_ns": duration(valid),
                "enabled_ns": duration(intersect(valid, record.enabled_intervals)),
                "active_ns": duration(active),
                "unavailable_ns": duration(unavailable),
                "denominator_reference": record.denominator_reference,
            }
        )
    for kind in ("command", "control"):
        targets = [e for e in data.targets if e.kind == kind]
        predictions = [e for e in data.predictions if e.kind == kind]
        backgrounds = {}
        for key, record in plans.items():
            excluded = [
                (
                    max(
                        record.start_ns,
                        min(e.start_ns, e.confirmation_ns + data.early_ns),
                    ),
                    min(record.end_ns, e.confirmation_ns + data.late_ns + 1),
                )
                for e in targets
                if (e.recording_id, e.timebase) == key
            ]
            backgrounds[key] = subtract(
                intersect(record.valid_intervals, record.non_operation_intervals),
                excluded,
            )
        background_ns = sum(duration(v) for v in backgrounds.values())
        layers = {}
        for layer, field in LAYERS.items():
            observed = [
                p
                for p in predictions
                if getattr(p, field) is not None
                and (layer == "submitted" or p.scope != "dry_run")
                and (layer != "application_local" or p.scope == "application-local")
                and (layer != "external_application" or p.scope == "external")
            ]
            valid_pred = [
                p
                for p in observed
                if contains(
                    plans[p.recording_id, p.timebase].valid_intervals, getattr(p, field)
                )
            ]
            grouped_truth, grouped_pred = defaultdict(list), defaultdict(list)
            for event in targets:
                grouped_truth[event.match_group].append(event)
            for event in valid_pred:
                grouped_pred[event.match_group].append(event)
            matched = []
            unmatched_ids, duplicate_ids, unmatched_truth_ids = [], [], []
            deltas = []
            latencies = {
                name: []
                for name in (
                    "source_to_prediction",
                    "source_to_injection",
                    "reference_to_injection",
                    "action_start_to_injection",
                    "injection_to_application",
                    "source_to_application",
                )
            }
            fp_background = 0
            for key in sorted(set(grouped_truth) | set(grouped_pred)):
                truth, pred, pairs, duplicate = match_events(
                    grouped_truth[key],
                    grouped_pred[key],
                    timestamp_field=field,
                    early_ns=data.early_ns,
                    late_ns=data.late_ns,
                )
                used = {j for _, j in pairs}
                used_truth = {i for i, _ in pairs}
                unmatched_truth_ids.extend(
                    [event.recording_id, event.timebase, event.event_id]
                    for i, event in enumerate(truth)
                    if i not in used_truth
                )
                for j, event in enumerate(pred):
                    if j not in used:
                        unmatched_ids.append(
                            [event.recording_id, event.timebase, event.event_id]
                        )
                        if j in duplicate:
                            duplicate_ids.append(
                                [event.recording_id, event.timebase, event.event_id]
                            )
                        if contains(
                            backgrounds[event.recording_id, event.timebase],
                            getattr(event, field),
                        ):
                            fp_background += 1
                for i, j in pairs:
                    gt, event = truth[i], pred[j]
                    delta = getattr(event, field) - gt.confirmation_ns
                    deltas.append(delta)
                    matched.append(
                        {
                            "recording_id": gt.recording_id,
                            "timebase": gt.timebase,
                            "target_id": gt.event_id,
                            "prediction_id": event.event_id,
                            "scope": event.scope,
                            "delta_ns": delta,
                        }
                    )
                    if event.source_ns is not None and event.prediction_ns is not None:
                        latencies["source_to_prediction"].append(
                            event.prediction_ns - event.source_ns
                        )
                    if layer == "external_injected":
                        if event.source_ns is not None:
                            latencies["source_to_injection"].append(
                                event.injected_ns - event.source_ns
                            )
                        latencies["reference_to_injection"].append(
                            event.injected_ns - gt.confirmation_ns
                        )
                        latencies["action_start_to_injection"].append(
                            event.injected_ns - gt.start_ns
                        )
                    if layer in {"application_local", "external_application"}:
                        if event.injected_ns is not None:
                            latencies["injection_to_application"].append(
                                event.application_ns - event.injected_ns
                            )
                        if event.source_ns is not None:
                            latencies["source_to_application"].append(
                                event.application_ns - event.source_ns
                            )
            tp, fp, fn = (
                len(matched),
                len(valid_pred) - len(matched),
                len(targets) - len(matched),
            )
            hours = background_ns / 3_600_000_000_000
            layers[layer] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "matched": matched,
                "unmatched_targets": unmatched_truth_ids,
                "unmatched_predictions": unmatched_ids,
                "duplicate_predictions": duplicate_ids,
                "not_observed_at_layer": len(predictions) - len(observed),
                "outside_valid_predictions": len(observed) - len(valid_pred),
                "background_false_events": fp_background,
                "background_valid_ns": background_ns,
                "background_false_per_hour": fp_background / hours if hours else None,
                "zero_count_poisson_upper95_per_hour": -math.log(0.05) / hours
                if hours and fp_background == 0
                else None,
                "poisson_assumption": "stationary independent events; not a guarantee of long-term false-trigger rate",
                "match_timing": _summary(deltas),
                "latency": {name: _summary(v) for name, v in latencies.items()},
            }
        report["kinds"][kind] = {
            "targets": len(targets),
            "predictions": len(predictions),
            "scope_counts": {
                scope: sum(p.scope == scope for p in predictions)
                for scope in ("dry_run", "application-local", "external")
            },
            "external_execution_status_counts": {
                s: sum(p.external_injection_status == s for p in predictions)
                for s in ("not_attempted", "confirmed", "failed", "uncertain")
            },
            "layers": layers,
        }
    return report
