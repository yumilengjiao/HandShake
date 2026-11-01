"""No-install/no-network handoff tests; wheels/ELF bytes are procedural fixtures."""

import json
import subprocess
import sys
import zipfile

import pytest

from algorithm.bundle import digest, json_bytes
from backend.handoff import build_kit, inspect_kit, main, safe_path, wheel_metadata


def wheel(path, name="handshake-core", extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            name.replace("-", "_") + "-0.2.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.2.0\nRequires-Python: >=3.10\n",
        )
        if extra:
            z.writestr(extra, b"synthetic-only")
    return path


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    core = wheel(source / "handshake_core-0.2.0-py3-none-any.whl")
    lock = source / "uv.lock"
    lock.write_text("version = 1\n")
    license = source / "LICENSE"
    license.write_text("Synthetic license fixture.\n")
    return dict(
        core_wheel=core,
        lockfile=lock,
        license_file=license,
        code_revision="a" * 40,
        review_reference="synthetic-fixture-not-release",
    )


def repin(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    for name, entry in manifest["files"].items():
        raw = (root / name).read_bytes()
        entry["bytes"] = len(raw)
        entry["sha256"] = digest(raw)
    raw = json_bytes(manifest)
    path.write_bytes(raw)
    return digest(raw)


def test_core_kit_roundtrip_no_absolute_paths_no_false_installer_claim(
    inputs, tmp_path
):
    out = tmp_path / "kit"
    receipt = build_kit(out, **inputs)
    report = inspect_kit(out, expected_sha256=receipt["manifest_sha256"])
    assert (
        report["status"] == "integrity_verified"
        and not report["launch_verified"]
        and not report["actions_enabled"]
    )
    assert report["dependency_resolution"] == "not_verified"
    assert str(tmp_path) not in (out / "manifest.json").read_text()
    assert not receipt["runtime_bundled"]
    with pytest.raises(FileExistsError):
        build_kit(out, **inputs)


def test_explicit_dependency_wheels_only(inputs, tmp_path):
    wh = tmp_path / "wheels"
    wh.mkdir()
    wheel(wh / "numpy.whl", "numpy")
    receipt = build_kit(tmp_path / "kit", wheelhouse=wh, **inputs)
    assert (
        inspect_kit(tmp_path / "kit", expected_sha256=receipt["manifest_sha256"])[
            "files"
        ]
        == 5
    )
    (wh / "session.html").write_text("private")
    with pytest.raises(ValueError, match="wheels"):
        build_kit(tmp_path / "bad", wheelhouse=wh, **inputs)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../key",
        "/abs",
        "foo/../secret",
        "foo//bar",
        ".env",
        "foo/.git/config",
        "C:secret",
        "a\\b",
    ],
)
def test_portable_relative_paths_reject_traversal_and_hidden_files(name):
    with pytest.raises(ValueError):
        safe_path(name)


@pytest.mark.parametrize(
    "member",
    [
        "../credential",
        "/absolute",
        "backend/.env",
        "private/records.json",
        "pi-session.html",
    ],
)
def test_bad_core_wheel_members_fail_before_output(inputs, tmp_path, member):
    wheel(inputs["core_wheel"], extra=member)
    with pytest.raises(ValueError):
        build_kit(tmp_path / "bad", **inputs)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "extra",
        "missing",
        "symlink",
        "executable",
        "rehashed_private",
        "overclaim",
        "target",
    ],
)
def test_modified_or_overstated_kit_is_not_accepted(inputs, tmp_path, problem):
    root = tmp_path / "kit"
    receipt = build_kit(root, **inputs)
    pin = receipt["manifest_sha256"]
    if problem == "hash":
        (root / "LICENSE").write_text("changed")
    if problem == "extra":
        (root / "session.html").write_text("not allowed")
    if problem == "missing":
        (root / "LICENSE").unlink()
    if problem == "symlink":
        (root / "LICENSE").unlink()
        (root / "LICENSE").symlink_to(inputs["license_file"])
    if problem == "executable":
        (root / "LICENSE").chmod(0o755)
    if problem in {"rehashed_private", "overclaim", "target"}:
        data = json.loads((root / "manifest.json").read_text())
        if problem == "rehashed_private":
            (root / "session.html").write_text("private")
            data["files"]["session.html"] = {
                "bytes": 7,
                "sha256": digest(b"private"),
                "executable": False,
            }
        elif problem == "target":
            data["target"]["secret"] = "unreviewed"
        else:
            data["launch_verified"] = True
        raw = json_bytes(data)
        (root / "manifest.json").write_bytes(raw)
        pin = digest(raw)
    with pytest.raises((ValueError, OSError)):
        inspect_kit(root, expected_sha256=pin)


def test_hash_pin_and_rehashed_dependency_identity_are_verified(inputs, tmp_path):
    root = tmp_path / "kit"
    build_kit(root, **inputs)
    with pytest.raises(ValueError, match="SHA256"):
        inspect_kit(root, expected_sha256="0" * 64)
    wheel(root / "wheels" / inputs["core_wheel"].name, name="wrong-core")
    with pytest.raises(ValueError, match="core wheel"):
        inspect_kit(root, expected_sha256=repin(root))


def test_native_allowlist_modes_and_platform_mismatch(inputs, tmp_path):
    native = tmp_path / "native"
    for name in (
        "handshake_stage",
        "lib/libapp.so",
        "lib/libflutter_linux_gtk.so",
        "lib/native_assets.json",
        "data/icudtl.dat",
        "data/flutter_assets/NOTICES.Z",
    ):
        path = native / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"\x7fELF synthetic fixture" if name == "handshake_stage" else b"synthetic"
        )
    root = tmp_path / "kit"
    result = build_kit(root, native=native, python_version="3.10", **inputs)
    report = inspect_kit(root, expected_sha256=result["manifest_sha256"])
    assert not report["launch_verified"]
    if sys.version_info[:2] != (3, 10):
        assert not report["platform_python_match"]
    assert (root / "native/handshake_stage").stat().st_mode & 0o111
    (native / "data/participants.csv").write_text("private")
    with pytest.raises(ValueError, match="allowlist"):
        build_kit(tmp_path / "bad", native=native, **inputs)


def test_model_requires_pin_and_license_review(inputs, tmp_path):
    with pytest.raises(ValueError, match="model requires"):
        build_kit(tmp_path / "bad", model="unused", **inputs)
    assert not (tmp_path / "bad").exists()


def test_cli_inspection_never_installs_launches_or_imports_torch(
    inputs, tmp_path, capsys
):
    root = tmp_path / "kit"
    result = build_kit(root, **inputs)
    assert (
        main(["inspect", "--input", str(root), "--sha256", result["manifest_sha256"]])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "integrity_verified"
    code = "import sys; from backend.handoff import main; assert 'torch' not in sys.modules and 'cv2' not in sys.modules; main()"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "inspect",
            "--input",
            str(root),
            "--sha256",
            result["manifest_sha256"],
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_model_copy_is_integrity_only_and_pin_is_enforced(inputs, tmp_path):
    from algorithm.bundle import MEMBERS

    source = tmp_path / "model"
    source.mkdir()
    members = {
        name: b"UNTRAINED FORMAT FIXTURE; NOT USABLE WEIGHTS" for name in MEMBERS
    }
    for name, raw in members.items():
        (source / name).write_bytes(raw)
    raw = json_bytes(
        {
            "format_version": "inference-bundle-v1",
            "status": "uncalibrated_preview_only",
            "files": {
                name: {"sha256": digest(value), "bytes": len(value)}
                for name, value in members.items()
            },
        }
    )
    (source / "checksums.json").write_bytes(raw)
    result = build_kit(
        tmp_path / "kit",
        model=source,
        model_sha256=digest(raw),
        model_license="synthetic-fixture-license-review",
        **inputs,
    )
    assert (
        inspect_kit(tmp_path / "kit", expected_sha256=result["manifest_sha256"])[
            "launch_verified"
        ]
        is False
    )
    with pytest.raises(ValueError, match="SHA256"):
        build_kit(
            tmp_path / "bad",
            model=source,
            model_sha256="0" * 64,
            model_license="synthetic",
            **inputs,
        )
    assert not (tmp_path / "bad").exists()


def test_wheel_duplicates_and_core_substitution_are_rejected(inputs, tmp_path):
    with zipfile.ZipFile(inputs["core_wheel"], "a") as z:
        z.writestr("handshake_core-0.2.0.dist-info/METADATA", "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        wheel_metadata(inputs["core_wheel"])
