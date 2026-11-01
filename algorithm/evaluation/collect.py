"""Read-only batched model scoring. No optimizer, checkpoint loading or actions."""

import re
from pathlib import Path

import numpy as np

from algorithm.bundle import load_bundle, read_bounded, strict_json
from algorithm.data.cache import load_cache
from algorithm.data.dataset import canonical_digest
from algorithm.evaluation.scores import (
    MAX_ROWS,
    ScoreProvenance,
    endpoint_digest,
    score_dtype,
    write_scores,
)


def collect_scores(
    package,
    cache,
    output,
    *,
    model_sha256,
    cache_index_sha256,
    split,
    allow_synthetic=False,
    batch_size=256,
    threads=1,
):
    if split not in {"val", "test"}:
        raise ValueError("scoring requires val or test split")
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError("score batch size must be in [1,1024]")
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("threads must be in [1,8]")
    if not isinstance(cache_index_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", cache_index_sha256
    ):
        raise ValueError("trusted cache index SHA256 required")
    if Path(output).exists():
        raise FileExistsError("score archives are immutable")
    root = Path(cache)
    if root.is_symlink():
        raise ValueError("linked cache directory")
    header = strict_json(read_bounded(root / "index.json", 32 * 1024**2))
    if (
        header.get("index_sha256") != cache_index_sha256
        or canonical_digest({k: v for k, v in header.items() if k != "index_sha256"})
        != cache_index_sha256
    ):
        raise ValueError("cache index does not match trusted receipt")
    bundle = load_bundle(package, expected_manifest_sha256=model_sha256)
    if (header.get("synthetic") or bundle.synthetic_training) and not allow_synthetic:
        raise ValueError("synthetic scoring requires explicit opt-in")
    # Validation inventory must be the one recorded at checkpoint selection.
    training = bundle.training_identity["data"]
    if split == "val" and training["val_index_sha256"] != cache_index_sha256:
        raise ValueError("validation cache differs from model selection inventory")
    if header.get("synthetic") is not bundle.synthetic_training:
        raise ValueError("cache/model synthetic provenance mismatch")
    arrays, index = load_cache(
        root,
        expected_dataset_sha256=bundle.normalizer.stats.dataset_hash,
        expected_split_sha256=bundle.normalizer.stats.split_hash,
        expected_stats_sha256=bundle.normalizer.stats_hash,
        expected_config_sha256=bundle.normalizer.config.config_hash,
        expected_split=split,
        expected_vocabulary=bundle.vocabulary,
    )
    if index["index_sha256"] != cache_index_sha256:
        raise ValueError("cache index changed during loading")
    count = len(arrays["y"])
    if count > MAX_ROWS:
        raise ValueError("score row budget exceeded")
    provenance = ScoreProvenance(
        model_sha256=model_sha256,
        dataset_sha256=index["dataset_sha256"],
        split_sha256=index["split_sha256"],
        stats_sha256=index["stats_sha256"],
        config_sha256=index["config_sha256"],
        cache_index_sha256=cache_index_sha256,
        endpoint_sha256=endpoint_digest(index),
        vocabulary_version=bundle.vocabulary.vocabulary_version,
        split=split,
        data_synthetic=index["synthetic"],
        model_synthetic=bundle.synthetic_training,
    )
    import torch

    torch.set_num_threads(threads)
    bundle.model.eval()

    def batches():
        # Score all represented endpoints (including partial windows), just like
        # checkpoint evaluation. Ready is a reported covariate, never a truth mask.
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                selected = slice(start, min(start + batch_size, count))
                x = torch.from_numpy(
                    np.array(arrays["x"][selected], dtype=np.float32, copy=True)
                )
                logits = (
                    bundle.model(x[:, -1] if bundle.model_type == "mlp" else x)
                    .cpu()
                    .numpy()
                )
                rows = np.zeros(
                    len(x), dtype=score_dtype(bundle.vocabulary.class_count)
                )
                if logits.shape != rows["logits"].shape:
                    raise ValueError("model logits shape mismatch")
                rows["label"], rows["class_valid"] = (
                    arrays["y"][selected],
                    arrays["class_valid"][selected],
                )
                rows["ready"], rows["available"], rows["logits"] = (
                    arrays["ready"][selected],
                    1,
                    logits,
                )
                yield rows
            # Compact cache maps are immutable throughout use, including the last batch.
            if hasattr(arrays["x"], "verify"):
                arrays["x"].verify()

    return write_scores(output, provenance, batches())
