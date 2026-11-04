"""Single-process trainer with explicit CPU/CUDA/MPS placement and epoch-boundary resume."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
from uuid import uuid4

import numpy as np
import torch
import yaml

from algorithm.data.dataset import canonical_digest
from algorithm.model_config import build_model, model_architecture
from algorithm.models.losses import masked_cross_entropy
from algorithm.training.balancing import class_weights_from_counts
from algorithm.training.config import TrainConfig
from algorithm.training.devices import device_identity, resolve_device
from algorithm.training.inputs import load_training_inputs
from algorithm.training.metrics import metrics_from_confusion
from algorithm.training.sampling import training_population
from shared.classes import vocabulary_for
from shared.features import feature_schema_hash


def _cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unreported"


def implementation_hash():
    root = Path(__file__).resolve().parents[2]
    files = []
    for package in ("algorithm", "shared"):
        files.extend(
            path
            for path in (root / package).rglob("*")
            if path.suffix in {".py", ".json", ".yaml"}
        )
    return canonical_digest(
        {
            path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        }
    )


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state(generator, device="cpu"):
    state = np.random.get_state()
    result = dict(
        python=random.getstate(),
        numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
        torch=torch.get_rng_state(),
        sampler=generator.get_state(),
    )
    if device.startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _evaluate(model, data, kind, batch_size):
    device = next(model.parameters()).device
    model.eval()
    matrix = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data["y"]), batch_size):
            selected = slice(start, start + batch_size)
            x = torch.from_numpy(data["x"][selected]).to(
                device=device, dtype=torch.float32
            )
            y = torch.from_numpy(
                np.array(data["y"][selected], dtype=np.int64, copy=True)
            ).to(device)
            mask = torch.from_numpy(
                np.array(data["class_valid"][selected], dtype=bool, copy=True)
            ).to(device)
            logits = model(x[:, -1] if kind == "mlp" else x)
            loss = masked_cross_entropy(logits, y, mask)
            valid = int(mask.sum())
            if loss is not None:
                total_loss += float(loss) * valid
                count += valid
                np.add.at(
                    matrix,
                    (y[mask].cpu().numpy(), logits.argmax(-1)[mask].cpu().numpy()),
                    1,
                )
    return metrics_from_confusion(matrix) | {"loss": total_loss / count}


def train(
    train_cache,
    val_cache,
    output,
    *,
    config=None,
    resume=None,
    stop_after_epoch=None,
    progress=None,
):
    if os.environ.get("HANDSHAKE_NO_TRAINING") == "1":
        raise RuntimeError(
            "HANDSHAKE_NO_TRAINING forbids neural training in this check process"
        )
    config = config or TrainConfig()
    device = resolve_device(config.device)
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int or stop_after_epoch < 1
    ):
        raise ValueError("stop_after_epoch must be a positive epoch boundary")
    training, validation, provenance = load_training_inputs(train_cache, val_cache)
    compact = provenance.get("format_version") == "t11-cache-v3"
    if compact and config.batch_size > 1024:
        raise ValueError("compact cache batch size must not exceed 1024")
    population = training_population(training["class_valid"], config.sampling)
    weights = class_weights_from_counts(
        provenance["train"]["class_counts"], config.class_weighting
    )
    class_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if weights is not None
        else None
    )
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    if config.shuffle_labels_diagnostic:
        training["y"] = np.array(training["y"], copy=True)
        mask = training["class_valid"]
        training["y"][mask] = np.random.default_rng(config.seed).permutation(
            training["y"][mask]
        )
    vocabulary = vocabulary_for(provenance["vocabulary_version"])
    model = build_model(
        config.model_type, vocabulary=vocabulary, variant=config.model_variant
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_factor,
            patience=config.lr_patience,
            threshold=config.min_delta,
            threshold_mode="abs",
            min_lr=config.min_learning_rate,
            eps=0.0,
        )
        if config.scheduler == "plateau"
        else None
    )
    identity = dict(
        config=config.model_dump(mode="json"),
        sampling=dict(
            strategy=config.sampling,
            replacement=False,
            endpoints_per_epoch=len(population),
            excluded_unsupervised=provenance["train"]["endpoints"] - len(population),
            population_sha256=sha256(population.tobytes()).hexdigest(),
            population_source=(
                "train.class_valid endpoint IDs only; no ready filtering"
                if config.sampling == "supervised_uniform"
                else "all train endpoint IDs, including unsupervised rows"
            ),
            validation_population="every supervised endpoint, unchanged",
        ),
        objective=dict(
            loss="masked_cross_entropy",
            class_weighting=config.class_weighting,
            class_weights=weights,
            counts_source="train.class_valid endpoints only",
            normalization="train expected weight one; mean over supervised batch rows",
            validation_loss="unweighted cross entropy over every supervised endpoint",
        ),
        data=provenance,
        code_sha256=implementation_hash(),
        schema_sha256=feature_schema_hash(),
        classes_sha256=vocabulary.digest(),
        classes=vocabulary.to_dict(),
        architecture=model_architecture(
            config.model_type, vocabulary=vocabulary, variant=config.model_variant
        ),
        environment=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            system=platform.system(),
            kernel=platform.release(),
            machine=platform.machine(),
            cpu_model=_cpu_model(),
            logical_cpus=os.cpu_count(),
            python_build=list(platform.python_build()),
            threads=config.threads,
            **device_identity(device),
        ),
    )
    identity_hash = canonical_digest(identity)
    root = Path(output)
    start, global_step, best_score, best_epoch, bad_epochs, history = (
        0,
        0,
        -1.0,
        0,
        0,
        [],
    )
    best_state = None
    if resume is not None:
        # Only this trainer's trusted local artifacts, never arbitrary downloaded pickle.
        if Path(resume).resolve() != (root / "last.pt").resolve():
            raise ValueError("resume must be this run directory's last.pt")
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        expected_hash = manifest.get("artifacts", {}).get("last.pt")
        if (
            expected_hash is not None
            and sha256(Path(resume).read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("resume checkpoint hash mismatch")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("format_version") != "trainer-checkpoint-v1"
            or checkpoint.get("identity_sha256") != identity_hash
        ):
            raise ValueError(
                "resume config/data/code/environment identity mismatch; start a new run"
            )
        if (
            manifest["run_id"] != checkpoint["run_id"]
            or manifest["identity"] != identity
        ):
            raise ValueError("resume run manifest mismatch")
        run_id = checkpoint["run_id"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        _restore_rng(checkpoint["rng"], generator)
        start, global_step = checkpoint["epoch"], checkpoint["global_step"]
        best_score, best_epoch, bad_epochs = (
            checkpoint["best_score"],
            checkpoint["best_epoch"],
            checkpoint["bad_epochs"],
        )
        history, best_state = checkpoint["history"], checkpoint["best_state_dict"]
    else:
        root.mkdir(parents=True, exist_ok=False)
        run_id = str(uuid4())
        (root / "config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
    manifest = dict(
        format_version="training-run-v1",
        run_id=run_id,
        identity_sha256=identity_hash,
        identity=identity,
        synthetic=provenance["synthetic"],
        status="running",
        resume_boundary="completed_epoch",
        determinism=f"{device}, strict deterministic algorithms, one process, no DataLoader workers; resume requires identical code/software/hardware; unsupported accelerator ops fail rather than silently fall back",
        selection=f"validation macro-F1 over all {vocabulary.class_count} classes; strict improvement > min_delta; ties retain earlier epoch",
        scheduler=(
            "ReduceLROnPlateau on validation macro-F1; state saved at epoch boundary"
            if scheduler is not None
            else "constant; no scheduler state"
        ),
        final_test_used=False,
    )
    _atomic_json(root / "run_manifest.json", manifest)
    try:
        for epoch in range(start + 1, config.max_epochs + 1):
            if bad_epochs >= config.patience:
                break
            for data in (training, validation):
                if "_verify" in data:
                    data["_verify"]()
            model.train()
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            order = population[
                torch.randperm(len(population), generator=generator).numpy()
            ]
            epoch_start_step = global_step
            total_loss, supervised, skipped, grad_max = 0.0, 0, 0, 0.0
            batch_metrics = []

            def record_batch(value):
                if not compact or len(batch_metrics) < 256:
                    batch_metrics.append(value)

            for batch_index, offset in enumerate(
                range(0, len(order), config.batch_size)
            ):
                selected = order[offset : offset + config.batch_size]
                x = torch.from_numpy(training["x"][selected]).to(
                    device=device, dtype=torch.float32
                )
                y = torch.from_numpy(
                    np.array(training["y"][selected], dtype=np.int64, copy=True)
                ).to(device)
                mask = torch.from_numpy(
                    np.array(training["class_valid"][selected], dtype=bool, copy=True)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[:, -1] if config.model_type == "mlp" else x)
                loss = masked_cross_entropy(
                    logits, y, mask, class_weights=class_weights
                )
                valid = int(mask.sum())
                if loss is None:
                    skipped += 1
                    record_batch(
                        dict(
                            batch=batch_index,
                            supervised=0,
                            skipped=True,
                            loss=None,
                            gradient_norm=None,
                            learning_rate=epoch_learning_rate,
                            class_counts=[0] * vocabulary.class_count,
                        )
                    )
                    continue
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                if any(
                    not torch.isfinite(parameter).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError("nonfinite parameters after optimizer update")
                global_step += 1
                supervised += valid
                total_loss += float(loss.detach()) * valid
                grad_max = max(grad_max, float(grad))
                record_batch(
                    dict(
                        batch=batch_index,
                        supervised=valid,
                        skipped=False,
                        loss=float(loss.detach()),
                        gradient_norm=float(grad),
                        learning_rate=epoch_learning_rate,
                        class_counts=[
                            int(((y == i) & mask).sum())
                            for i in range(vocabulary.class_count)
                        ],
                    )
                )
            metrics = _evaluate(model, validation, config.model_type, config.batch_size)
            for data in (training, validation):
                if "_verify" in data:
                    data["_verify"]()
            score = metrics["macro_f1"]
            if best_state is None or score > best_score + config.min_delta:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
            else:
                bad_epochs += 1
            if scheduler is not None:
                scheduler.step(score)
            history.append(
                dict(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=total_loss / supervised,
                    train_loss_kind=(
                        "sqrt_inverse_weighted_ce"
                        if weights is not None
                        else "unweighted_ce"
                    ),
                    supervised=supervised,
                    sampling=config.sampling,
                    sampled_endpoints=len(order),
                    unsupervised_sampled=len(order) - supervised,
                    optimizer_steps=global_step - epoch_start_step,
                    skipped_batches=skipped,
                    gradient_norm_max=grad_max,
                    learning_rate=epoch_learning_rate,
                    next_learning_rate=optimizer.param_groups[0]["lr"],
                    batches=batch_metrics,
                    validation=metrics,
                    **(
                        {
                            "batch_detail_limit": 256,
                            "batches_total": len(
                                range(0, len(order), config.batch_size)
                            ),
                        }
                        if compact
                        else {}
                    ),
                )
            )
            payload = dict(
                format_version="trainer-checkpoint-v1",
                run_id=run_id,
                identity_sha256=identity_hash,
                identity=identity,
                epoch=epoch,
                global_step=global_step,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                rng=_rng_state(generator, device),
                history=history,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                best_state_dict=best_state,
            )
            _atomic_checkpoint(root / "last.pt", payload)
            _atomic_checkpoint(
                root / "best.pt",
                dict(
                    format_version="trainer-best-v1",
                    run_id=run_id,
                    identity=identity,
                    epoch=best_epoch,
                    val_macro_f1=best_score,
                    state_dict=best_state,
                ),
            )
            temporary = root / "epochs.jsonl.tmp"
            temporary.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in history),
                encoding="utf-8",
            )
            os.replace(temporary, root / "epochs.jsonl")
            if progress is not None:
                progress(history[-1])
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
        # Recreate best after a crash between last and best writes, even if already early-stopped.
        if best_state is not None:
            _atomic_checkpoint(
                root / "best.pt",
                dict(
                    format_version="trainer-best-v1",
                    run_id=run_id,
                    identity=identity,
                    epoch=best_epoch,
                    val_macro_f1=best_score,
                    state_dict=best_state,
                ),
            )
        # The checkpoint is authoritative after interruption during log publication.
        temporary = root / "epochs.jsonl.tmp"
        temporary.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in history),
            encoding="utf-8",
        )
        os.replace(temporary, root / "epochs.jsonl")
        completed = bool(
            history
            and (
                history[-1]["epoch"] >= config.max_epochs
                or bad_epochs >= config.patience
            )
        )
        manifest.update(
            status="completed" if completed else "paused_at_epoch",
            epochs=len(history),
            global_step=global_step,
            best_epoch=best_epoch,
            best_val_macro_f1=best_score,
            early_stopped=bad_epochs >= config.patience,
            artifacts={
                name: sha256((root / name).read_bytes()).hexdigest()
                for name in ("last.pt", "best.pt", "epochs.jsonl")
            },
        )
        _atomic_json(root / "run_manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="failed", failure_type=type(error).__name__, failure=str(error)
        )
        _atomic_json(root / "run_manifest.json", manifest)
        raise
