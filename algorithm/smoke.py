"""Small deterministic tensor memorization check, NOT a gesture experiment.

No checkpoint is saved. This is only T12/T13 optimization/shape diagnostics;
formal datasets, validation selection, event metrics and training belong to T14+.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from algorithm.model_config import build_model, load_model_config
from algorithm.models.rule_baseline import RuleBaseline, RuleThresholds
from shared.classes import load_classes
from shared.features import FeatureConfig, feature_schema_hash


def _metrics(targets, predictions, tokens):
    confusion = np.zeros((len(tokens), len(tokens)), dtype=int)
    for truth, prediction in zip(targets, predictions):
        confusion[int(truth), int(prediction)] += 1
    rows = {}
    for index, token in enumerate(tokens):
        correct = int(confusion[index, index])
        support, predicted = int(confusion[index].sum()), int(confusion[:, index].sum())
        precision, recall = correct / predicted if predicted else 0.0, correct / support if support else 0.0
        rows[token] = dict(support=support, precision=precision, recall=recall,
                           f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return dict(confusion_matrix=confusion.tolist(), per_class=rows,
                macro_f1=sum(row["f1"] for row in rows.values()) / len(tokens))


def _rule_case(pose, displacement):
    points = np.zeros((21, 3))
    for finger, mcp in enumerate((5, 9, 13, 17)):
        x = (finger - 1.5) * 0.3
        points[mcp], points[mcp + 1] = (x, 1, 0), (x, 1.5, 0)
        folded = pose == "fist" or (pose == "neutral" and finger == 0)
        points[mcp + 2] = (x + 0.2, 1.25, 0) if folded else (x, 2, 0)
        points[mcp + 3] = (x + 0.1, 1.1, 0) if folded else (x, 2.5, 0)
    values = np.zeros((32, 94))
    values[:, :63] = points.ravel()
    values[:, 63] = np.linspace(0.3, 0.3 + displacement, 32)
    values[:, 64], values[:, 65] = 0.4, displacement / (31 / 30)
    values[:, 67], values[:, 69] = 0.1, 1 / 30
    values[:, 71:] = 1.0
    return values, np.arange(32, dtype=np.int64) * 33_333_333


def run_smoke(*, seed=42, steps=100) -> dict:
    import torch
    from algorithm.models.losses import masked_cross_entropy

    if type(seed) is not int or seed < 0 or type(steps) is not int or steps <= 0:
        raise ValueError("seed and steps must be nonnegative/positive integers")
    torch.manual_seed(seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocabulary = load_classes()
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(vocabulary.class_count).repeat_interleave(4)
    # Easily separable, deliberately artificial class codes. Not anatomical data.
    inputs = torch.randn((len(targets), 32, 94), generator=generator) * 0.01
    for index, target in enumerate(targets):
        inputs[index, :, int(target)] += 2.0
    inputs[:, :, 70] = 0.0
    inputs[:, :, 71:] = 1.0
    mask = torch.ones(len(targets), dtype=torch.bool)
    report = dict(scope="synthetic_tensor_memorization_and_rule_fixtures_only", seed=seed, cpu_threads=1,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  feature_schema_version="landmarks94-v1", schema_sha256=feature_schema_hash(),
                  classes_sha256=vocabulary.digest(), classes=list(vocabulary.tokens),
                  synthetic_input_sha256=sha256(inputs.numpy().tobytes()).hexdigest(),
                  dataset_version=None, checkpoint_saved=False, formal_validation_performed=False,
                  feature_config=FeatureConfig().to_dict(), models={})
    try:
        for kind in ("mlp", "tcn"):
            torch.manual_seed(seed)
            model = build_model(kind)
            data = inputs[:, -1, :] if kind == "mlp" else inputs
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            losses = []
            started = time.perf_counter()
            model.train()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = masked_cross_entropy(model(data), targets, mask)
                if loss is None:
                    continue
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ValueError("nonfinite synthetic gradients")
                optimizer.step()
                losses.append(loss.item())
            training_ms = (time.perf_counter() - started) * 1000
            model.eval()
            with torch.no_grad():
                predicted = model(data).argmax(-1).numpy()
                for _ in range(5):
                    model(data[:1])
                durations = []
                for _ in range(30):
                    start = time.perf_counter_ns()
                    model(data[:1])
                    durations.append((time.perf_counter_ns() - start) / 1e6)
            report["models"][kind] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                         receptive_field=getattr(model, "receptive_field", None),
                                         config=load_model_config(kind).model_dump(mode="json"),
                                         samples=len(targets), updates=len(losses),
                                         first_training_loss=losses[0], last_training_loss=losses[-1],
                                         training_ms=training_ms,
                                         cpu_forward_ms=dict(p50=float(np.percentile(durations, 50)),
                                                             p95=float(np.percentile(durations, 95)), repeats=30),
                                         memorized_all=bool(np.array_equal(predicted, targets.numpy())),
                                         synthetic_training_metrics=_metrics(targets.numpy(), predicted, vocabulary.tokens))
        rule = RuleBaseline()
        expected = [0, 1, 2, 3, 4]
        cases = [("neutral", 0), ("open", 0), ("open", -0.4), ("open", 0.4), ("fist", 0)]
        actual = [vocabulary.id_for(rule.classify(*_rule_case(pose, dx))) for pose, dx in cases]
        report["models"]["rule"] = dict(parameters=0, history_frames=32, thresholds=asdict(RuleThresholds()),
                                         synthetic_fixture_metrics=_metrics(expected, actual, vocabulary.tokens))
    finally:
        torch.set_num_threads(previous_threads)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    report = run_smoke(seed=args.seed, steps=args.steps)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    print(encoded, end="")
    return int(not all(report["models"][kind]["memorized_all"] for kind in ("mlp", "tcn")))


if __name__ == "__main__":
    raise SystemExit(main())
