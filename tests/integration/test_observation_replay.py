"""Observation replay uses actual untrained bundle forward passes, never a trainer."""

import json
import subprocess
import sys

import pytest

from algorithm.bundle import digest, json_bytes
from algorithm.replay import (
    ReplayArchive,
    ReplayFrame,
    ReplayItem,
    json_bytes_compact,
    main,
    run_replay,
)
from tests.integration.test_evaluation import scoring_fixture as _scoring_fixture
from tests.integration.test_evaluation import zero_bundle
from tests.integration.test_feature_dataset import example_records


@pytest.fixture(params=["legacy", "compact"])
def scoring_fixture(request, tmp_path):
    return _scoring_fixture.__wrapped__(request, tmp_path)


def items():
    values = [ReplayItem(kind="start", at_ns=0)]
    for row in example_records(subjects=("A",), count=40):
        values.append(
            ReplayItem(
                kind="frame",
                at_ns=row.frame_status.source_timestamp_ns,
                frame=ReplayFrame(
                    width=640,
                    height=480,
                    transform=row.observation.transform_metadata,
                    status=row.frame_status,
                    observations=(row.observation,),
                ),
            )
        )
    values += [
        ReplayItem(kind="tick", at_ns=2_000_000_000),
        ReplayItem(kind="stop", at_ns=2_000_000_001),
    ]
    return values


def archive(root, values=None, **changes):
    root.mkdir()
    raw = b"".join(
        json_bytes_compact(v.model_dump(mode="json"))
        for v in (values if values is not None else items())
    )
    (root / "events.jsonl").write_bytes(raw)
    manifest = (
        dict(
            format_version="observation-replay-v1",
            synthetic=True,
            review_reference="procedural-test-NOT-real-data",
            stream_id="s",
            timebase="synthetic",
            events=len(raw.splitlines()),
            events_bytes=len(raw),
            events_sha256=digest(raw),
        )
        | changes
    )
    data = json_bytes(manifest)
    (root / "manifest.json").write_bytes(data)
    return digest(data)


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_real_bundle_replay_is_deterministic_streaming_and_deadline_driven(
    scoring_fixture, kind, monkeypatch
):
    import torch

    root, normalizer, indexes = scoring_fixture
    model = zero_bundle(root / "bundle", kind, normalizer, indexes["val"])
    sha = archive(root / "input")

    def forbidden(*a, **k):
        pytest.fail("replay must not train or unpickle")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    for name in ("first", "second"):
        report = run_replay(
            root / "input",
            root / name,
            manifest_sha256=sha,
            package=root / "bundle",
            model_sha256=model,
            allow_synthetic=True,
        )
        assert (
            report["complete"]
            and not report["actions_enabled"]
            and report["frames"] == 40
        )
        assert report["counts"]["frame:predicted"] == 9
        assert report["counts"]["timeout:input_stalled"] == 1
    assert (root / "first/predictions.jsonl").read_bytes() == (
        root / "second/predictions.jsonl"
    ).read_bytes()
    rows = [
        json.loads(line)
        for line in (root / "first/predictions.jsonl").read_text().splitlines()
    ]
    timeout = next(r for r in rows if r["kind"] == "timeout")
    assert timeout["at_ns"] == 39 * 30_000_000 + normalizer.config.max_age_ns + 1
    assert all("inference_ns" not in row for row in rows)


def test_stop_reset_clear_history_without_relabelling_and_no_hand_is_explicit(
    scoring_fixture,
):
    root, normalizer, indexes = scoring_fixture
    model = zero_bundle(root / "bundle", "mlp", normalizer, indexes["val"])
    values = items()
    values.insert(36, ReplayItem(kind="reset", at_ns=1_020_000_000))
    payload = values[38].frame
    values[38] = ReplayItem(
        kind="frame",
        at_ns=values[38].at_ns,
        frame=ReplayFrame(
            width=640,
            height=480,
            transform=payload.transform,
            status=type(payload.status).model_validate(
                payload.status.model_dump() | {"perception_status": "processed_no_hand"}
            ),
            observations=(),
        ),
    )
    sha = archive(root / "input", values)
    report = run_replay(
        root / "input",
        root / "out",
        manifest_sha256=sha,
        package=root / "bundle",
        model_sha256=model,
        allow_synthetic=True,
    )
    assert report["counts"]["frame:predicted"] == 4
    rows = [
        json.loads(line)
        for line in (root / "out/predictions.jsonl").read_text().splitlines()
    ]
    assert any(r.get("reason") == "no_hand" and not r["logits"] for r in rows)
    assert any(
        r.get("context_epoch") == "2" and r["state"] == "warming_up" for r in rows
    )


@pytest.mark.parametrize(
    "problem",
    [
        "time",
        "duplicate",
        "stream",
        "count",
        "lifecycle",
        "synthetic",
        "truncated",
        "tamper",
        "symlink",
    ],
)
def test_invalid_archives_are_rejected_before_inference(tmp_path, problem):
    values = items()
    changes = {}
    if problem == "time":
        values[3] = values[1]
    if problem == "duplicate":
        values[2] = values[1]
    if problem == "stream":
        changes["stream_id"] = "other"
    if problem == "count":
        changes["events"] = 999
    if problem == "lifecycle":
        values.insert(1, ReplayItem(kind="start", at_ns=0))
    if problem == "synthetic":
        changes["synthetic"] = False
    sha = archive(tmp_path / "input", values, **changes)
    path = tmp_path / "input/events.jsonl"
    if problem == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    if problem == "tamper":
        path.write_bytes(path.read_bytes().replace(b'"start"', b'"stop "', 1))
    if problem == "symlink":
        path.rename(tmp_path / "linked")
        path.symlink_to(tmp_path / "linked")
    with pytest.raises(ValueError):
        ReplayArchive(tmp_path / "input", expected_sha256=sha)


def test_immutable_guard_and_wrong_pin(tmp_path):
    sha = archive(tmp_path / "input")
    with pytest.raises(ValueError, match="SHA256"):
        ReplayArchive(tmp_path / "input", expected_sha256="0" * 64)
    value = ReplayArchive(tmp_path / "input", expected_sha256=sha)
    with (tmp_path / "input/events.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        list(value.items())


def test_inspection_and_help_do_not_import_neural_runtime(tmp_path, capsys):
    sha = archive(tmp_path / "input")
    assert main(["inspect", "--input", str(tmp_path / "input"), "--sha256", sha]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 40
    code = "import sys; from algorithm.replay import main; assert 'torch' not in sys.modules and 'cv2' not in sys.modules; main(['--help'])"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0


def test_opt_in_and_lifecycle_failure_do_not_create_output(scoring_fixture):
    root, normalizer, indexes = scoring_fixture
    model = zero_bundle(root / "bundle", "mlp", normalizer, indexes["val"])
    sha = archive(root / "input")
    with pytest.raises(ValueError, match="opt-in"):
        run_replay(
            root / "input",
            root / "out",
            manifest_sha256=sha,
            package=root / "bundle",
            model_sha256=model,
        )
    assert not (root / "out").exists()
