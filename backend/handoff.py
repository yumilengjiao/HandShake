"""Draft Linux offline handoff kits. Explicit inputs, bounded copies, no install/launch/network."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import stat
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from algorithm.bundle import MEMBERS, digest, json_bytes, read_bounded, strict_json
from algorithm.data.source_archive import file_sha256

MAX_FILES = 10_000
MAX_FILE = 512 * 1024**2
MAX_TOTAL = 2 * 1024**3
HEX = re.compile(r"^[a-f0-9]{64}$")


def safe_path(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("invalid kit path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or str(path) != name
        or any(p.startswith(".") or p in {"", ".."} for p in path.parts)
    ):
        raise ValueError("hidden or traversing kit path")
    return path


def allowed_member(name):
    safe_path(name)
    if name in {"LICENSE", "OFFLINE_README.md", "environment/uv.lock"}:
        return True
    if name.startswith("wheels/"):
        return len(PurePosixPath(name).parts) == 2 and name.endswith(".whl")
    if name.startswith("model/"):
        return name[6:] in MEMBERS | {"checksums.json"}
    if name.startswith("native/"):
        part = name[7:]
        return (
            part == "handshake_stage"
            or part.startswith("lib/")
            and part.endswith(".so")
            or part in {"data/icudtl.dat", "lib/native_assets.json"}
            or part.startswith("data/flutter_assets/")
            and (
                PurePosixPath(part).suffix.lower()
                in {
                    ".json",
                    ".bin",
                    ".otf",
                    ".ttf",
                    ".frag",
                    ".spv",
                    ".dat",
                    ".txt",
                    ".z",
                    ".png",
                    ".jpg",
                    ".webp",
                }
                or PurePosixPath(part).name == "NOTICES"
            )
        )
    return False


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def regular(path, limit=MAX_FILE):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("artifact symlinks are forbidden")
    s = path.stat()
    if not stat.S_ISREG(s.st_mode) or not 0 < s.st_size <= limit:
        raise ValueError("artifact must be a nonempty bounded regular file")
    return s


def wheel_metadata(path, *, core=False):
    regular(path)
    if path.suffix != ".whl":
        raise ValueError("wheelhouse may contain only explicitly reviewed wheels")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 30000 or sum(i.file_size for i in members) > MAX_TOTAL:
            raise ValueError("expanded wheel budget exceeded")
        names = [i.filename for i in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel members")
        for item in members:
            safe_path(item.filename.rstrip("/"))
            if stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
                raise ValueError("linked/encrypted wheel member")
            if core and not item.filename.startswith(
                ("algorithm/", "backend/", "shared/", "handshake_core-")
            ):
                raise ValueError("core wheel contains unexpected top-level content")
        metadata = [i for i in members if i.filename.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or metadata[0].file_size > 1_048_576:
            raise ValueError("one bounded wheel METADATA required")
        value = BytesParser().parsebytes(archive.read(metadata[0]))
    if (
        not value["Name"]
        or not value["Version"]
        or core
        and value["Name"].replace("_", "-").lower() != "handshake-core"
    ):
        raise ValueError("wrong wheel identity")
    return {
        "name": value["Name"],
        "version": value["Version"],
        "requires_python": value["Requires-Python"],
        "requires_dist": value.get_all("Requires-Dist", []),
    }


def model_files(directory, pin):
    root = Path(directory)
    if not isinstance(pin, str) or not HEX.fullmatch(pin):
        raise ValueError("trusted model SHA256 required")
    if root.is_symlink() or {p.name for p in root.iterdir()} != MEMBERS | {
        "checksums.json"
    }:
        raise ValueError("model inventory mismatch")
    raw = read_bounded(root / "checksums.json", 1_048_576)
    if digest(raw) != pin:
        raise ValueError("model manifest SHA256 mismatch")
    manifest = strict_json(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != "inference-bundle-v1"
        or manifest.get("status") != "uncalibrated_preview_only"
        or set(manifest["files"]) != MEMBERS
    ):
        raise ValueError("unsupported model handoff format")
    for name in MEMBERS:
        data = read_bounded(
            root / name, 4_194_304 if name.endswith(".npz") else 1_048_576
        )
        if (
            len(data) != manifest["files"][name]["bytes"]
            or digest(data) != manifest["files"][name]["sha256"]
        ):
            raise ValueError("model member hash mismatch")
    return [
        (root / name, "model/" + name, False)
        for name in sorted(MEMBERS | {"checksums.json"})
    ]


def native_files(directory):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("native directory must not be a symlink")
    required = {
        "handshake_stage",
        "lib/libapp.so",
        "lib/libflutter_linux_gtk.so",
        "data/icudtl.dat",
    }
    result = []
    for index, path in enumerate(root.rglob("*")):
        if index >= MAX_FILES * 2:
            raise ValueError("native inventory scan budget exceeded")
        relative = path.relative_to(root).as_posix()
        safe_path(relative)
        if path.is_symlink():
            raise ValueError("native symlinks must be resolved during reviewed build")
        if path.is_dir():
            continue
        allowed = (
            relative == "handshake_stage"
            or relative.startswith("lib/")
            and path.suffix == ".so"
            or relative in {"data/icudtl.dat", "lib/native_assets.json"}
            or relative.startswith("data/flutter_assets/")
            and (
                path.suffix.lower()
                in {
                    ".json",
                    ".bin",
                    ".otf",
                    ".ttf",
                    ".frag",
                    ".spv",
                    ".dat",
                    ".txt",
                    ".z",
                    ".png",
                    ".jpg",
                    ".webp",
                }
                or path.name == "NOTICES"
            )
        )
        if not allowed:
            raise ValueError("native file is not on the release allowlist: " + relative)
        regular(path)
        result.append((path, "native/" + relative, relative == "handshake_stage"))
    if not required.issubset({p.relative_to(root).as_posix() for p, _, _ in result}):
        raise ValueError("incomplete Linux release bundle")
    with (root / "handshake_stage").open("rb") as executable:
        if executable.read(4) != b"\x7fELF":
            raise ValueError("Linux ELF executable required")
    return sorted(result, key=lambda item: item[1])


INSTRUCTIONS = """# HandShake draft offline handoff

Integrity-only staging kit, NOT an installer, embedded Python runtime, signed release,
or certification of model accuracy, license compliance or camera/application permissions.
No training, installation, network access or executable launch occurs during build/inspect.

1. Obtain manifest SHA256 from the trusted sender OUTSIDE this directory.
2. Install Python matching manifest.target.python and a packaging tool separately.
3. Verify with `python -m backend.handoff inspect --input KIT --sha256 TRUSTED_SHA`.
4. In a NEW virtual environment, use pip's offline resolver before installing:
   `python -m pip install --dry-run --ignore-installed --no-index --find-links KIT/wheels CORE_WHEEL`
   Then repeat without --dry-run only after reviewing the plan. Missing wheels are a blocker,
   not permission to silently download packages. The lockfile is an audit reference;
   this kit does not claim the arbitrary supplied wheelhouse resolves that lock exactly.
5. Run `python -m backend.interaction.public_replay --allow-synthetic --output NEW_DIRECTORY`
   for the included code's procedural no-camera/no-training smoke. Not recognition evidence.
6. If native/ is included, run native/handshake_stage only after system-library review,
   with HANDSHAKE_PYTHON pointing to the new environment. Clear HANDSHAKE_MODEL_PACKAGE,
   HANDSHAKE_MODEL_SHA256, HANDSHAKE_PERCEPTION_MODEL, HANDSHAKE_CALIBRATION and
   HANDSHAKE_CALIBRATION_SHA256 unless separately reviewed assets are explicitly chosen.
   Set HANDSHAKE_ROOT to an existing neutral working directory, not a private source path.
7. Model assets are preview-only. Their byte integrity is checked, not forward correctness.
   Verify the actual model with handshake-model inspect and the externally recorded model SHA.

Keep each kit immutable. Roll back by selecting a previously verified kit/environment;
do not mix wheel/model/native files between revisions. Delete a kit directory to remove
that copy only: this does not delete original datasets, environments, backups or trained data.
No credentials, session HTML, raw recordings or private identity maps are intentionally collected.
Allowlisting is NOT a content/DLP or legal audit; review-reference records the operator's review.
"""


def build_kit(
    output,
    *,
    core_wheel,
    lockfile,
    license_file,
    code_revision,
    review_reference,
    python_version=None,
    wheelhouse=None,
    native=None,
    model=None,
    model_sha256=None,
    model_license=None,
):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("handoff kits are immutable")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", code_revision)
        or not isinstance(review_reference, str)
        or not review_reference.strip()
        or len(review_reference) > 256
    ):
        raise ValueError("full source revision and explicit review reference required")
    python_version = (
        python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not re.fullmatch(r"3\.(10|11|12|13|14)", python_version):
        raise ValueError("supported Python major/minor required")
    if (
        bool(model) != bool(model_sha256)
        or model
        and (not model_license or len(model_license) > 256)
    ):
        raise ValueError("model requires a trusted hash and license review reference")
    wheel = Path(core_wheel)
    core = wheel_metadata(wheel, core=True)
    files = [
        (wheel, "wheels/" + wheel.name, False),
        (Path(lockfile), "environment/uv.lock", False),
        (Path(license_file), "LICENSE", False),
    ]
    regular(Path(lockfile), 2_097_152)
    regular(Path(license_file), 131_072)
    dependencies = []
    if wheelhouse:
        folder = Path(wheelhouse)
        if folder.is_symlink():
            raise ValueError("linked wheelhouse")
        for path in sorted(folder.iterdir()):
            if path.resolve() == wheel.resolve():
                continue
            metadata = wheel_metadata(path)
            if metadata["name"].replace("_", "-").lower() == "handshake-core":
                raise ValueError("wheelhouse contains another core distribution")
            dependencies.append(metadata)
            files.append((path, "wheels/" + path.name, False))
    if native:
        if destination.resolve().is_relative_to(Path(native).resolve()):
            raise ValueError("output cannot be nested inside native source")
        files.extend(native_files(native))
    if model:
        files.extend(model_files(model, model_sha256))
    if len(files) > MAX_FILES or len({name for _, name, _ in files}) != len(files):
        raise ValueError("duplicate/oversized kit inventory")
    total = 0
    stamps = {}
    for path, name, _ in files:
        safe_path(name)
        total += regular(path).st_size
        stamps[path] = stamp(path)
    if total > MAX_TOTAL:
        raise ValueError("kit byte budget exceeded")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for path, name, executable in files:
        if stamp(path) != stamps[path]:
            raise ValueError("source changed during staging")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before_hash = file_sha256(path)
        with path.open("rb") as source, target.open("xb") as out:
            copied = 0
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > stamps[path][2]:
                    raise ValueError("source grew during copy")
                out.write(chunk)
        if stamp(path) != stamps[path] or file_sha256(target) != before_hash:
            raise ValueError("source changed during copy")
        target.chmod(0o755 if executable else 0o644)
        inventory[name] = {
            "sha256": before_hash,
            "bytes": copied,
            "executable": executable,
        }
    if model:
        model_files(destination / "model", model_sha256)
    instructions = INSTRUCTIONS.encode()
    (destination / "OFFLINE_README.md").write_bytes(instructions)
    inventory["OFFLINE_README.md"] = {
        "sha256": digest(instructions),
        "bytes": len(instructions),
        "executable": False,
    }
    manifest = {
        "format_version": "offline-handoff-v1",
        "status": "draft_integrity_only",
        "complete": True,
        "code_revision": code_revision,
        "review_reference": review_reference,
        "target": {"os": "linux", "machine": "x86_64", "python": python_version},
        "core": core,
        "dependencies": dependencies,
        "dependency_resolution": "not_verified",
        "runtime_bundled": False,
        "launch_verified": False,
        "native_included": bool(native),
        "model_sha256": model_sha256,
        "model_license_reference": model_license,
        "scope": "read_only_preview_and_synthetic_replay",
        "files": inventory,
    }
    raw = json_bytes(manifest)
    (destination / "manifest.json").write_bytes(raw)  # publish last
    return {
        "status": "kit_staged",
        "manifest_sha256": digest(raw),
        "files": len(inventory),
        "bytes": sum(v["bytes"] for v in inventory.values()),
        "dependency_resolution": "not_verified",
        "runtime_bundled": False,
        "launch_verified": False,
    }


def inspect_kit(directory, *, expected_sha256):
    root = Path(directory)
    if (
        root.is_symlink()
        or not isinstance(expected_sha256, str)
        or not HEX.fullmatch(expected_sha256)
    ):
        raise ValueError("regular kit and external manifest SHA256 required")
    raw = read_bounded(root / "manifest.json", 4_194_304)
    if digest(raw) != expected_sha256:
        raise ValueError("kit manifest SHA256 mismatch")
    value = strict_json(raw)
    expected = {
        "format_version",
        "status",
        "complete",
        "code_revision",
        "review_reference",
        "target",
        "core",
        "dependencies",
        "dependency_resolution",
        "runtime_bundled",
        "launch_verified",
        "native_included",
        "model_sha256",
        "model_license_reference",
        "scope",
        "files",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["format_version"] != "offline-handoff-v1"
        or value["complete"] is not True
        or value["status"] != "draft_integrity_only"
        or value["scope"] != "read_only_preview_and_synthetic_replay"
        or value["dependency_resolution"] != "not_verified"
        or value["runtime_bundled"] is not False
        or value["launch_verified"] is not False
    ):
        raise ValueError("unsupported or overstated handoff manifest")
    if (
        not isinstance(value["code_revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", value["code_revision"])
        or not isinstance(value["review_reference"], str)
        or not value["review_reference"].strip()
        or not isinstance(value["dependencies"], list)
        or not all(isinstance(v, dict) for v in value["dependencies"])
    ):
        raise ValueError("invalid handoff provenance")
    inventory = value["files"]
    if (
        not isinstance(inventory, dict)
        or not 1 <= len(inventory) <= MAX_FILES
        or not {"LICENSE", "OFFLINE_README.md", "environment/uv.lock"}.issubset(
            inventory
        )
    ):
        raise ValueError("invalid kit inventory")
    actual = set()
    allowed_dirs = {
        str(parent)
        for name in inventory
        for parent in safe_path(name).parents
        if str(parent) != "."
    }
    for index, path in enumerate(root.rglob("*")):
        if index >= MAX_FILES * 2:
            raise ValueError("kit inventory scan budget exceeded")
        if path.is_symlink():
            raise ValueError("kit contains a symlink")
        if path.is_dir() and path.relative_to(root).as_posix() not in allowed_dirs:
            raise ValueError("unexpected kit directory")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory) | {"manifest.json"}:
        raise ValueError("kit inventory mismatch")
    total = 0
    for name, entry in inventory.items():
        safe_path(name)
        if not isinstance(entry, dict):
            raise ValueError("invalid kit checksum entry")
        if (
            not allowed_member(name)
            or entry.get("executable") is True
            and name != "native/handshake_stage"
        ):
            raise ValueError("kit contains a disallowed artifact or executable")
        if (
            set(entry) != {"sha256", "bytes", "executable"}
            or type(entry["bytes"]) is not int
            or not 0 < entry["bytes"] <= MAX_FILE
            or type(entry["executable"]) is not bool
            or not isinstance(entry["sha256"], str)
            or not HEX.fullmatch(entry["sha256"])
        ):
            raise ValueError("invalid kit checksum entry")
        info = regular(root / name)
        before = stamp(root / name)
        if (
            info.st_size != entry["bytes"]
            or file_sha256(root / name) != entry["sha256"]
            or stamp(root / name) != before
        ):
            raise ValueError("kit member hash/size changed")
        if bool(info.st_mode & 0o111) != entry["executable"]:
            raise ValueError("kit executable mode mismatch")
        total += info.st_size
        if total > MAX_TOTAL:
            raise ValueError("kit byte budget exceeded")
    wheels = [root / name for name in inventory if name.startswith("wheels/")]
    metadata = [(path, wheel_metadata(path)) for path in wheels]
    core_wheels = [
        (path, item)
        for path, item in metadata
        if item["name"].replace("_", "-").lower() == "handshake-core"
    ]
    if len(core_wheels) != 1 or core_wheels[0][1] != value["core"]:
        raise ValueError("core wheel metadata mismatch")
    wheel_metadata(core_wheels[0][0], core=True)
    if sorted(
        (item for path, item in metadata if path != core_wheels[0][0]),
        key=lambda v: (v["name"], v["version"]),
    ) != sorted(value["dependencies"], key=lambda v: (v["name"], v["version"])):
        raise ValueError("dependency wheel metadata mismatch")
    if type(value["native_included"]) is not bool or value["native_included"] != any(
        name.startswith("native/") for name in inventory
    ):
        raise ValueError("native inventory flag mismatch")
    if value["native_included"]:
        native_files(root / "native")
    if value["model_sha256"] is not None:
        model_files(root / "model", value["model_sha256"])
    elif any(name.startswith("model/") for name in inventory):
        raise ValueError("unbound model in kit")
    target = value["target"]
    if set(target) != {"os", "machine", "python"}:
        raise ValueError("invalid target profile")
    compatible = target == {
        "os": platform.system().lower(),
        "machine": platform.machine().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    installed = {}
    for name in ("numpy", "pydantic", "PyYAML"):
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {
        "status": "integrity_verified",
        "manifest_sha256": expected_sha256,
        "files": len(inventory),
        "bytes": total,
        "target": target,
        "platform_python_match": compatible,
        "installed_core_dependencies": installed,
        "dependency_resolution": "not_verified",
        "launch_verified": False,
        "actions_enabled": False,
        "next_step": "offline resolver in a fresh environment, then explicit smoke; no automatic install or launch",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in (
        "output",
        "core-wheel",
        "lockfile",
        "license-file",
        "code-revision",
        "review-reference",
    ):
        build.add_argument("--" + name, required=True)
    for name in (
        "python-version",
        "wheelhouse",
        "native",
        "model",
        "model-sha256",
        "model-license",
    ):
        build.add_argument("--" + name)
    check = commands.add_parser("inspect")
    check.add_argument("--input", required=True)
    check.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_kit(args.input, expected_sha256=args.sha256)
        else:
            kwargs = vars(args).copy()
            kwargs.pop("command")
            output = kwargs.pop("output")
            result = build_kit(output, **kwargs)
        print(json_bytes(result).decode(), end="")
        return 0
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"handoff rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
