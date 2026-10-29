"""Training-only raw-observation augmentation; derivatives are recomputed later.

Time reversal/speed changes are deliberately absent until their label semantics
are validated. This function never mutates the source recording.
"""

import numpy as np

from algorithm.data.dataset import ObservationRecord, records_digest
from shared.classes import load_classes


def augment_training_records(records: list[ObservationRecord], *, source_split: str,
                             seed: int, noise_std: float = 0.0,
                             perception_dropout: float = 0.0,
                             horizontal_flip: bool = False) -> list[ObservationRecord]:
    if source_split != "train":
        raise ValueError("augmentation is training-only")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(horizontal_flip) is not bool:
        raise ValueError("horizontal_flip must be bool")
    if (type(noise_std) not in (int, float) or not np.isfinite(noise_std) or noise_std < 0
            or type(perception_dropout) not in (int, float) or not np.isfinite(perception_dropout) or not 0 <= perception_dropout <= 1):
        raise ValueError("invalid augmentation magnitudes")
    rng, vocabulary = np.random.default_rng(seed), load_classes()
    parent_hash = records_digest(records)
    swapped = {vocabulary.id_for("swipe_left"): vocabulary.id_for("swipe_right"),
               vocabulary.id_for("swipe_right"): vocabulary.id_for("swipe_left")}
    result = []
    for record in records:
        data = record.model_dump(mode="json")
        if record.source_split not in (None, "train"):
            raise ValueError("cannot relabel a validation/test source as train")
        data["source_split"] = "train"
        data["parent_records_sha256"] = parent_hash
        data["augmentation"] = record.augmentation + [f"seed={seed}", f"noise_std={noise_std}",
                                                       f"perception_dropout={perception_dropout}",
                                                       f"horizontal_flip={horizontal_flip}"]
        if horizontal_flip:
            data["class_id"] = swapped.get(record.class_id, record.class_id)
        if record.observation is not None:
            points = np.asarray(record.observation.landmarks_image).copy()
            if horizontal_flip:
                points[:, 0] = 1.0 - points[:, 0]
            if noise_std:
                points += rng.normal(0, noise_std, points.shape)
            data["observation"]["landmarks_image"] = points.tolist()
            if perception_dropout and rng.random() < perception_dropout:
                data["observation"] = None
                data["frame_status"]["perception_status"] = "error"
                data["frame_status"]["drop_or_error_reason"] = "training_augmentation_perception_dropout"
        result.append(ObservationRecord.model_validate(data))
    return result
