"""Bind T04 whole-frame raw geometry to the exact T16 prediction source."""

from backend.interaction.engine import Evidence
from backend.interaction.geometry import frame_geometry
from shared.contracts import ModelPrediction


def evidence_from_frame(prediction, frame, *, ready):
    if not isinstance(prediction, ModelPrediction):
        raise ValueError(
            "source classification is not a legacy application gesture prediction"
        )
    info = frame.info
    if (
        prediction.stream_id,
        prediction.source_frame_id,
        prediction.source_timestamp_ns,
        prediction.timebase,
    ) != (info.stream_id, info.frame_id, info.timestamp_ns, info.timebase):
        raise ValueError("prediction and whole-frame evidence source mismatch")
    geometry = frame_geometry(frame, prediction.track_id)
    scores = (
        prediction.calibrated_probabilities
        if prediction.calibrated_probabilities is not None
        else prediction.logits
    )
    winner = max(range(5), key=scores.__getitem__)
    static = geometry.quality_valid and (
        (winner == 0 and geometry.neutral)
        or (winner == 1 and geometry.extended == 4)
        or (winner == 4 and geometry.folded == 4)
    )
    return Evidence(
        prediction,
        ready,
        geometry.quality_valid,
        geometry.visible_tracks,
        association_ambiguous=geometry.ambiguous,
        static_pose_valid=bool(static),
        neutral_valid=geometry.neutral,
        palm_xy=geometry.palm_xy,
        aspect_ratio=geometry.aspect_ratio,
    )
