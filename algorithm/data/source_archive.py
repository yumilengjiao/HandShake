"""Read-only, bounded ZIP access after caller-supplied SHA256 verification.

No extraction, scripts, network requests, or replacement of expected checksums.
"""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text):
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=reject)


class VerifiedArchive:
    """Limits apply to declared and actual member size; no paths are extracted."""

    def __init__(
        self,
        path,
        *,
        expected_sha256,
        max_member_bytes=64 * 1024 * 1024,
        max_total_bytes=8 * 1024**3,
        max_members=10000,
    ):
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise ValueError("an explicit lowercase SHA256 is required")
        for limit in (max_member_bytes, max_total_bytes, max_members):
            if type(limit) is not int or limit <= 0:
                raise ValueError("archive limits must be positive integers")
        self.path = Path(path)
        self.max_member_bytes = max_member_bytes
        self._stream = self.path.open("rb")
        self.zip = None
        try:
            digest = sha256()
            for block in iter(lambda: self._stream.read(1024 * 1024), b""):
                digest.update(block)
            self.digest = digest.hexdigest()
            if self.digest != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch: expected {expected_sha256}, observed {self.digest}"
                )
            self._stream.seek(0)
            # Use the same open file for hashing and reading, not a second path lookup.
            self.zip = zipfile.ZipFile(self._stream)
            entries = self.zip.infolist()
            if (
                len(entries) > max_members
                or sum(i.file_size for i in entries) > max_total_bytes
            ):
                raise ValueError("archive total size/member limit exceeded")
            self.members = {}
            for entry in entries:
                name = entry.filename
                parts = PurePosixPath(name).parts
                mode = entry.external_attr >> 16
                if (
                    not parts
                    or name.startswith("/")
                    or "\\" in name
                    or ":" in name
                    or ".." in parts
                    or "\x00" in name
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError("unsafe archive member path/type")
                if name in self.members:
                    raise ValueError("duplicate archive member")
                if entry.flag_bits & 1 or entry.file_size > max_member_bytes:
                    raise ValueError("encrypted or oversized archive member")
                self.members[name] = entry
        except BaseException:
            if self.zip is not None:
                self.zip.close()
            self._stream.close()
            raise

    def find(self, suffix):
        matches = [
            name
            for name, info in self.members.items()
            if not info.is_dir() and (name == suffix or name.endswith("/" + suffix))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one archive member: {suffix}")
        return matches[0]

    def read_bytes(self, name):
        entry = self.members[name]
        with self.zip.open(entry) as stream:
            content = stream.read(self.max_member_bytes + 1)
        if len(content) > self.max_member_bytes or len(content) != entry.file_size:
            raise ValueError("archive member actual size mismatch/limit exceeded")
        return content

    def read_text(self, name):
        return self.read_bytes(name).decode("utf-8-sig")

    def read_json(self, name):
        return strict_json(self.read_text(name))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()
        self._stream.close()
