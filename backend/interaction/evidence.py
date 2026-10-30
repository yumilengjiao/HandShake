"""Bind T04 whole-frame raw geometry to the exact T16 prediction source."""
import math
import numpy as np

from backend.interaction.engine import Evidence
from shared.config import load_config
from shared.contracts import ModelPrediction


def evidence_from_frame(prediction, frame, *, ready):
    if not isinstance(prediction, ModelPrediction):
        raise ValueError("source classification is not a legacy application gesture prediction")
    info = frame.info
    if (prediction.stream_id, prediction.source_frame_id, prediction.source_timestamp_ns, prediction.timebase) != (
        info.stream_id, info.frame_id, info.timestamp_ns, info.timebase
    ):
        raise ValueError("prediction and whole-frame evidence source mismatch")
    tracks = tuple(observation.track_id for observation in frame.observations)
    observation = next((o for o in frame.observations if o.track_id == prediction.track_id), None)
    ambiguous = any("association" in flag or "ambiguous" in flag for o in frame.observations for flag in o.quality_flags)
    valid = observation is not None and observation.observation_valid and observation.quality >= 0.5 and all(observation.validity_mask)
    neutral, static, center = False, False, None
    if valid:
        protocol = load_config().gestures
        points = np.asarray(observation.landmarks_image, dtype=np.float64)
        center = tuple(float(value) for value in points[[0, 5, 9, 13, 17], :2].mean(0))
        corrected = points.copy()
        corrected[:, 1] *= info.height / info.width
        angles = []
        for mcp, pip, dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
            a, b = corrected[mcp] - corrected[pip], corrected[dip] - corrected[pip]
            length = np.linalg.norm(a) * np.linalg.norm(b)
            angles.append(None if length <= 1e-12 else math.degrees(math.acos(float(np.clip(np.dot(a, b) / length, -1, 1)))))
        extended = sum(a is not None and a >= protocol.extended_pip_degrees_min for a in angles)
        folded = sum(a is not None and a <= protocol.folded_pip_degrees_max for a in angles)
        xmin, xmax, ymin, ymax = protocol.neutral_region_xy
        neutral = 1 <= extended <= 3 and extended + folded == 4 and xmin <= center[0] <= xmax and ymin <= center[1] <= ymax
        scores = prediction.calibrated_probabilities if prediction.calibrated_probabilities is not None else prediction.logits
        winner = max(range(5), key=scores.__getitem__)
        static = (winner == 0 and neutral) or (winner == 1 and extended == 4) or (winner == 4 and folded == 4)
    return Evidence(prediction, ready, valid, tracks, association_ambiguous=ambiguous,
                    static_pose_valid=bool(static), neutral_valid=bool(neutral), palm_xy=center, aspect_ratio=info.height / info.width)
