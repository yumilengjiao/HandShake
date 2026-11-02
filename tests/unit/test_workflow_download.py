"""Downloader fake responses: no network or user data required."""

import hashlib
import io

import pytest

from backend.workflows.download import HTTPSRedirects, _download_url, fetch
from backend.workflows.workspace import WorkflowError


class Response(io.BytesIO):
    def __init__(self, data, **headers):
        super().__init__(data)
        self.headers = {"Content-Type": "application/zip", **headers}


def entry(data=b"PK-synthetic-archive"):
    return {
        "url": "https://drive.google.com/file/d/abc_123/view",
        "expected_sha256": hashlib.sha256(data).hexdigest(),
    }


def test_stream_download_is_pinned_idempotent_and_keeps_no_partial(tmp_path):
    target = tmp_path / "input.zip"
    body = b"PK-synthetic-archive"
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        return Response(body, **{"Content-Length": str(len(body))})

    fetch(entry(body), target, limit=100, opener=opener)
    fetch(
        entry(body), target, limit=100, opener=lambda *a, **k: pytest.fail("redownload")
    )
    assert target.read_bytes() == body
    assert "drive.usercontent.google.com" in calls[0]
    assert not target.with_suffix(".zip.part").exists()


def test_mismatched_upstream_is_quarantined_not_published(tmp_path):
    target = tmp_path / "input.zip"
    with pytest.raises(WorkflowError, match="SHA256"):
        fetch(
            entry(), target, limit=100, opener=lambda *a, **k: Response(b"PK-different")
        )
    assert not target.exists()
    assert len(list(tmp_path.glob("*.rejected-*.zip"))) == 1
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "body,headers,limit",
    [
        (b"<html>login</html>", {"Content-Type": "text/html"}, 100),
        (b"not-a-zip", {}, 100),
        (b"PK-too-large", {}, 3),
        (b"PK-incomplete", {"Content-Length": "90"}, 100),
        (b"", {}, 100),
    ],
)
def test_bad_download_never_becomes_formal_data(tmp_path, body, headers, limit):
    target = tmp_path / "input.zip"
    with pytest.raises(WorkflowError):
        fetch(
            entry(body),
            target,
            limit=limit,
            opener=lambda *a, **k: Response(body, **headers),
        )
    assert not target.exists() and not list(tmp_path.glob("*.part"))


def test_does_not_delete_another_process_partial(tmp_path):
    target = tmp_path / "input.zip"
    partial = target.with_suffix(".zip.part")
    partial.write_bytes(b"other process")
    with pytest.raises(WorkflowError, match="未完成"):
        fetch(entry(), target, limit=100)
    assert partial.read_bytes() == b"other process"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/data.zip",
        "file:///etc/passwd",
        "https://user:secret@example.org/data.zip",
    ],
)
def test_unsafe_source_urls_are_rejected(url):
    with pytest.raises(WorkflowError):
        _download_url(entry() | {"url": url})


def test_public_drive_confirmation_is_bounded_to_same_file_and_official_host(tmp_path):
    body = b"PK-synthetic-archive"
    html = b"""<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
    <input type="hidden" name="id" value="abc_123"><input type="hidden" name="export" value="download">
    <input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="public-token"></form>"""
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return (
            Response(html, **{"Content-Type": "text/html"})
            if len(calls) == 1
            else Response(body)
        )

    fetch(entry(body), tmp_path / "file.zip", limit=100, opener=opener)
    assert len(calls) == 2 and "confirm=t" in calls[1]


def test_redirect_cannot_downgrade_tls():
    with pytest.raises(WorkflowError):
        HTTPSRedirects().redirect_request(
            None, None, 302, "", {}, "http://example.org/data.zip"
        )
