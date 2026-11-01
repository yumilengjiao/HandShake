"""Finite, auditable validation-only scalar calibration. Never changes neural weights."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from algorithm.bundle import digest, json_bytes, read_bounded, strict_json
from algorithm.evaluation.classification import evaluate_classification
from algorithm.evaluation.scores import ScoreProvenance
from shared.desktop_protocol import WireModel


class Calibration(WireModel):
    format_version: Literal["validation-calibration-v1"] = "validation-calibration-v1"
    status: Literal["draft"] = "draft"
    source: ScoreProvenance
    score_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic: bool
    temperature: float = Field(ge=0.05, le=100, allow_inf_nan=False)
    temperatures: tuple[float, ...] = Field(min_length=1, max_length=64)
    nlls: tuple[float, ...] = Field(min_length=1, max_length=64)
    thresholds: tuple[float, ...] = Field(min_length=1, max_length=64)
    threshold: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    rejection_status: Literal["selected", "infeasible"]
    max_risk: float = Field(ge=0, le=1, allow_inf_nan=False)
    min_coverage: float = Field(gt=0, le=1, allow_inf_nan=False)
    min_accepted: int = Field(ge=1)
    labeled_count: int = Field(ge=1)
    fit_count: int = Field(ge=1)
    accepted_count: int = Field(ge=0)
    accepted_errors: int = Field(ge=0)
    actions_enabled: Literal[False] = False
    method: Literal["finite-grid-nll_then_max-coverage-empirical-risk-v1"] = (
        "finite-grid-nll_then_max-coverage-empirical-risk-v1"
    )

    @model_validator(mode="after")
    def consistent(self):
        import math

        if self.source.split != "val" or self.synthetic is not self.source.synthetic:
            raise ValueError("calibration must retain validation/synthetic provenance")
        if (
            len(self.temperatures) != len(self.nlls)
            or len(set(self.temperatures)) != len(self.temperatures)
            or 1.0 not in self.temperatures
            or self.temperature not in self.temperatures
            or any(
                not math.isfinite(v) or not 0.05 <= v <= 100 for v in self.temperatures
            )
            or any(not math.isfinite(v) or v < 0 for v in self.nlls)
            or len(set(self.thresholds)) != len(self.thresholds)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.thresholds)
        ):
            raise ValueError("invalid calibration search record")
        best_nll = min(self.nlls)
        best = min(
            (i for i, nll in enumerate(self.nlls) if nll <= best_nll + 1e-12),
            key=lambda i: (abs(math.log(self.temperatures[i])), self.temperatures[i]),
        )
        if self.temperature != self.temperatures[best]:
            raise ValueError("temperature does not match recorded NLL selection")
        if (
            not self.accepted_errors
            <= self.accepted_count
            <= self.fit_count
            <= self.labeled_count
        ):
            raise ValueError("invalid calibration sample counts")
        if self.rejection_status == "selected":
            if (
                self.threshold not in self.thresholds
                or self.accepted_count < self.min_accepted
                or self.accepted_count / self.labeled_count < self.min_coverage
                or self.accepted_errors / self.accepted_count > self.max_risk
            ):
                raise ValueError("selected threshold violates coverage/risk budget")
        elif self.threshold is not None or self.accepted_count or self.accepted_errors:
            raise ValueError(
                "infeasible selection must not pretend all-reject is a solution"
            )
        return self


def fit_calibration(
    scores,
    *,
    allow_synthetic=False,
    temperatures=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0),
    thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    max_risk=0.1,
    min_coverage=0.5,
    min_accepted=20,
):
    import math

    if scores.provenance.split != "val":
        raise ValueError(
            "test data cannot fit calibration or choose rejection thresholds"
        )
    if scores.manifest_sha256 is None:
        raise ValueError("calibration requires a verified score archive")
    if scores.provenance.synthetic and not allow_synthetic:
        raise ValueError("synthetic calibration requires explicit opt-in")
    if (
        not 1 <= len(temperatures) <= 64
        or len(set(temperatures)) != len(temperatures)
        or 1.0 not in temperatures
        or any(
            type(t) not in (int, float) or not math.isfinite(t) or not 0.05 <= t <= 100
            for t in temperatures
        )
        or not 1 <= len(thresholds) <= 64
        or len(set(thresholds)) != len(thresholds)
        or any(
            type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1
            for t in thresholds
        )
        or type(min_accepted) is not int
        or min_accepted < 1
        or type(max_risk) not in (int, float)
        or not math.isfinite(max_risk)
        or not 0 <= max_risk <= 1
        or type(min_coverage) not in (int, float)
        or not math.isfinite(min_coverage)
        or not 0 < min_coverage <= 1
    ):
        raise ValueError("invalid finite search/coverage/risk budget")
    results = [evaluate_classification(scores, temperature=t) for t in temperatures]
    if results[0]["available_labeled"] == 0:
        raise ValueError("no available labeled validation scores")
    best_nll = min(result["nll_available"] for result in results)
    best = min(
        (i for i, r in enumerate(results) if r["nll_available"] <= best_nll + 1e-12),
        key=lambda i: (abs(math.log(temperatures[i])), temperatures[i]),
    )
    temperature = float(temperatures[best])
    choices = []
    for threshold in thresholds:
        result = evaluate_classification(
            scores, temperature=temperature, threshold=threshold
        )
        if (
            result["accepted_labeled"] >= min_accepted
            and result["coverage"] >= min_coverage
            and result["selective_risk"] <= max_risk
        ):
            choices.append((result, float(threshold)))
    selected = (
        min(
            choices,
            key=lambda row: (
                -row[0]["accepted_labeled"],
                row[0]["accepted_errors"],
                row[1],
            ),
        )
        if choices
        else None
    )
    return Calibration(
        source=scores.provenance,
        score_manifest_sha256=scores.manifest_sha256,
        synthetic=scores.provenance.synthetic,
        temperature=temperature,
        temperatures=tuple(float(t) for t in temperatures),
        nlls=tuple(r["nll_available"] for r in results),
        thresholds=tuple(float(t) for t in thresholds),
        threshold=selected[1] if selected else None,
        rejection_status="selected" if selected else "infeasible",
        max_risk=float(max_risk),
        min_coverage=float(min_coverage),
        min_accepted=min_accepted,
        labeled_count=results[0]["labeled_endpoints"],
        fit_count=results[0]["available_labeled"],
        accepted_count=selected[0]["accepted_labeled"] if selected else 0,
        accepted_errors=selected[0]["accepted_errors"] if selected else 0,
    )


def save_calibration(value, path):
    data = json_bytes(value.model_dump(mode="json"))
    with Path(path).open("xb") as stream:
        stream.write(data)
    return digest(data)


def load_for_bundle(path, *, expected_sha256, bundle):
    """Load an offline sidecar against the verified bundle's exact validation inventory.

    This supplies probabilities, not action authority or a frozen event policy.
    The original preview bundle and its checksum inventory remain unchanged.
    """
    data = bundle.training_identity["data"]
    target = ScoreProvenance(
        model_sha256=bundle.manifest_sha256,
        dataset_sha256=bundle.normalizer.stats.dataset_hash,
        split_sha256=bundle.normalizer.stats.split_hash,
        stats_sha256=bundle.normalizer.stats_hash,
        config_sha256=bundle.normalizer.config.config_hash,
        cache_index_sha256=data["val_index_sha256"],
        endpoint_sha256=data["val"]["endpoint_sha256"],
        vocabulary_version=bundle.vocabulary.vocabulary_version,
        split="val",
        data_synthetic=data["synthetic"],
        model_synthetic=bundle.synthetic_training,
    )
    return load_calibration(path, expected_sha256=expected_sha256, target=target)


def load_calibration(path, *, expected_sha256, target):
    data = read_bounded(Path(path), 1_048_576)
    if digest(data) != expected_sha256:
        raise ValueError("calibration artifact hash mismatch")
    # Reject duplicate keys/nonfinite constants before schema validation.
    value = Calibration.model_validate_json(json_bytes(strict_json(data)))
    for field in (
        "model_sha256",
        "dataset_sha256",
        "split_sha256",
        "stats_sha256",
        "config_sha256",
        "vocabulary_version",
        "data_synthetic",
        "model_synthetic",
        "scoring_policy",
    ):
        if getattr(value.source, field) != getattr(target, field):
            raise ValueError("calibration/model/data binding mismatch: " + field)
    if target.split == "val" and (
        value.source.cache_index_sha256 != target.cache_index_sha256
        or value.source.endpoint_sha256 != target.endpoint_sha256
    ):
        raise ValueError("validation calibration endpoint inventory mismatch")
    return value
