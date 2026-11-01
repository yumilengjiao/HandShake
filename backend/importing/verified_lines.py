"""Single-pass verified gzip JSONL for imported public artifacts (no extraction)."""

import gzip
from hashlib import sha256
from pathlib import Path

from algorithm.data.source_archive import strict_json


class _HashingReader:
    def __init__(self, raw):
        self.raw = raw
        self.digest = sha256()

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        return data


def verified_json_lines(path, expected_sha256, *, max_lines, max_line_bytes=65536):
    """Hash the compressed bytes actually consumed, not a separate path read.

    Consumers must exhaust the iterator before publishing a derived artifact.
    Oversized lines/counts, CRC failures and trailing corruption are rejected.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular imported shard")
    count = 0
    with path.open("rb") as raw:
        source = _HashingReader(raw)
        with gzip.GzipFile(fileobj=source, mode="rb") as stream:
            while line := stream.readline(max_line_bytes + 1):
                count += 1
                if len(line) > max_line_bytes or count > max_lines:
                    raise ValueError("imported shard line/count budget exceeded")
                yield strict_json(line)
        if source.digest.hexdigest() != expected_sha256:
            raise ValueError("imported shard consumed-byte hash mismatch")
    if count != max_lines:
        raise ValueError("imported shard line count mismatch")
