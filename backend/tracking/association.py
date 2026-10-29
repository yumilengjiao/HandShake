"""Conservative whole-frame association, at most two hands."""
from dataclasses import dataclass
from itertools import product

import numpy as np

from backend.perception.quality import QualitySettings, temporal_flags


@dataclass(frozen=True)
class TrackingSettings:
    max_distance: float = .25
    max_shape_distance: float = 1.0
    ambiguity_margin: float = .05
    overlap_distance: float = .05
    lost_timeout_ns: int = 500_000_000

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("tracking thresholds must be positive finite draft values")


@dataclass
class Track:
    track_id: str
    geometry: object
    handedness: str
    timestamp_ns: int


class Association:
    def __init__(self, settings=TrackingSettings(), quality=QualitySettings()):
        self.settings, self.quality = settings, quality
        self.reset()

    def reset(self):
        self.stream = None
        self.next_id = 0
        self.tracks = {}
        self.last_ns = None

    def _new_id(self):
        self.next_id += 1
        return f"track-{self.next_id}"

    def associate(self, info, geometries, handedness):
        if self.stream != info.stream_id:
            self.reset()
            self.stream = info.stream_id
        if self.last_ns is not None and info.timestamp_ns <= self.last_ns:
            raise ValueError("association requires increasing source time")
        self.last_ns = info.timestamp_ns
        self.tracks = {k:t for k,t in self.tracks.items() if info.timestamp_ns - t.timestamp_ns <= self.settings.lost_timeout_ns}
        old = list(self.tracks.values())
        n = len(geometries)
        if n > 2 or len(handedness) != n:
            raise ValueError("association requires at most two hands")
        costs = {}
        for i, g in enumerate(geometries):
            if g.flags:
                continue
            for j, t in enumerate(old):
                distance = float(np.linalg.norm(g.center - t.geometry.center))
                shape = float(np.sqrt(np.mean((g.shape - t.geometry.shape)**2)))
                if distance > self.settings.max_distance or shape > self.settings.max_shape_distance:
                    continue
                scale = abs(np.log(max(g.scale, 1e-6) / max(t.geometry.scale, 1e-6)))
                hand = .03 if handedness[i] != "unknown" and t.handedness != "unknown" and handedness[i] != t.handedness else 0.
                costs[i,j] = distance + .1 * shape + .05 * scale + hand
        # New-track alternatives have a high cost; two near-equal existing matches
        # are a conflict, not a license to choose by detector array order.
        choices = []
        for assignment in product(range(-1, len(old)), repeat=n):
            used = [j for j in assignment if j >= 0]
            if len(set(used)) != len(used) or any((i,j) not in costs for i,j in enumerate(assignment) if j >= 0):
                continue
            score = sum(.8 if j < 0 else costs[i,j] for i,j in enumerate(assignment))
            choices.append((score, assignment))
        choices.sort()
        ambiguous = n == 2 and np.linalg.norm(geometries[0].center - geometries[1].center) < self.settings.overlap_distance
        if len(choices) > 1 and choices[1][0] - choices[0][0] < self.settings.ambiguity_margin:
            ambiguous = True
        assignment = choices[0][1] if choices else (-1,) * n
        if ambiguous:
            self.tracks.clear()
            assignment = (-1,) * n
        results = []
        updates = {}
        for i, g in enumerate(geometries):
            previous = old[assignment[i]] if assignment[i] >= 0 else None
            track_id = previous.track_id if previous else self._new_id()
            flags = list(g.flags)
            if ambiguous:
                flags.append("association_ambiguous")
            if previous:
                flags += temporal_flags(g, previous.geometry, info.timestamp_ns - previous.timestamp_ns, self.quality)
            if not flags:
                updates[track_id] = Track(track_id, g, handedness[i], info.timestamp_ns)
            results.append((track_id, list(dict.fromkeys(flags))))
        self.tracks.update(updates)
        # At most two recent hypotheses: do not grow an unbounded assignment
        # search when many unmatched/replacement hands pass through the frame.
        self.tracks = dict(sorted(self.tracks.items(), key=lambda item: item[1].timestamp_ns, reverse=True)[:2])
        return results
