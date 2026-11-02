"""Explicit public HTTPS downloads with bounded streaming and pinned hashes."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from algorithm.data.source_archive import file_sha256
from backend.workflows.workspace import (
    WorkflowError,
    operation_lock,
    read_json,
    write_json,
)

ANNOTATIONS = "IPN_HandS_annotations.zip"
LANDMARKS = "IPN_HandS_landmarks_MPHands.zip"
LIMITS = {ANNOTATIONS: 4 * 1024**2, LANDMARKS: 1024**3}


def catalog(ws):
    path = ws.path(ws.settings.data.catalog)
    if not path.is_file():
        raise WorkflowError(
            f"缺少数据来源清单：{path}",
            "使用完整源码 checkout，或在 handshake.toml 中指定经过审核的 catalog。",
        )
    value = read_json(path)
    sources = [v for v in value.get("datasets", []) if v.get("id") == "ipn-hands"]
    if (
        value.get("schema_version") != "public-dataset-sources-v1"
        or len(sources) != 1
        or sources[0].get("license", {}).get("id") != "CC-BY-4.0"
    ):
        raise WorkflowError("数据来源清单的版本/来源/许可不正确。")
    entries = sources[0]["files"]
    mapping = {v["filename"]: v for v in entries}
    if len(mapping) != len(entries):
        raise WorkflowError("来源清单包含重复文件。")
    for name in LIMITS:
        if name not in mapping or not re.fullmatch(
            r"[0-9a-f]{64}", str(mapping[name].get("expected_sha256", ""))
        ):
            raise WorkflowError(f"来源清单缺少可信 SHA256：{name}")
    return mapping


def check_archive(path, expected):
    if not path.is_file():
        raise WorkflowError(
            f"缺少数据包：{path}",
            "下一步：uv run download；也可以从官方分享页下载后放到上述位置。",
        )
    actual = file_sha256(path)
    if actual != expected:
        raise WorkflowError(
            f"数据校验未通过：{path.name}\n  预期 SHA256：{expected}\n  实际 SHA256：{actual}",
            "不能拿此文件训练。核对上游版本/联系发布者；不要把实际哈希直接改成预期值。详情：data/SOURCES.md。",
        )
    return actual


def _download_url(entry):
    url = entry.get("download_url") or entry["url"]
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise WorkflowError("只允许不含账户凭据的公开 HTTPS 数据地址。")
    if parsed.hostname == "drive.google.com":
        query = urllib.parse.parse_qs(parsed.query)
        match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
        identifier = query.get("id", [None])[0] or (match.group(1) if match else None)
        if not identifier or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise WorkflowError(
                "无法解析官方 Google Drive 文件地址。",
                f"请从分享页手动下载：{entry['url']}",
            )
        return (
            "https://drive.usercontent.google.com/download?"
            + urllib.parse.urlencode({"id": identifier, "export": "download"})
        )
    return url


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise WorkflowError("拒绝从 HTTPS 跳转到不安全的下载地址。")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class _DriveForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.action = None
        self.fields = {}
        self.invalid = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "form" and attrs.get("id") == "download-form":
            if self.action is not None:
                self.invalid = True
            self.active = attrs.get("method", "get").lower() == "get"
            self.action = attrs.get("action")
        if self.active and tag == "input" and attrs.get("type") == "hidden":
            name, value = attrs.get("name"), attrs.get("value", "")
            if (
                name not in {"id", "export", "confirm", "uuid"}
                or name in self.fields
                or len(value) > 1024
            ):
                self.invalid = True
            else:
                self.fields[name] = value

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def _open_archive(request, opener):
    response = opener(request, timeout=30)
    if "html" not in response.headers.get("Content-Type", "").lower():
        return response
    # Only a bounded public virus-scan confirmation form is supported. No login,
    # cookies, JavaScript, arbitrary form targets or authentication bypass.
    with response:
        html = response.read(256 * 1024 + 1)
    url = urllib.parse.urlparse(request.full_url)
    if len(html) <= 256 * 1024 and url.hostname == "drive.usercontent.google.com":
        form = _DriveForm()
        form.feed(html.decode("utf-8", errors="replace"))
        expected_id = urllib.parse.parse_qs(url.query).get("id", [None])[0]
        if (
            not form.invalid
            and form.action == "https://drive.usercontent.google.com/download"
            and expected_id
            and form.fields.get("id") == expected_id
            and form.fields.get("export") == "download"
            and form.fields.get("confirm")
        ):
            confirmed = urllib.request.Request(
                form.action + "?" + urllib.parse.urlencode(form.fields),
                headers=dict(request.header_items()),
            )
            return opener(confirmed, timeout=30)
    raise WorkflowError(
        "服务器返回了登录、限流或不支持的确认页面。",
        "请从 data/SOURCES.md 的官方分享页手动下载，之后同一命令会校验已有文件。",
    )


def fetch(entry, target, *, limit, emit=print, opener=None):
    expected = entry["expected_sha256"]
    if target.exists():
        check_archive(target, expected)
        emit(f"已存在且校验通过，跳过：{target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise WorkflowError(
            f"发现未完成下载：{partial}",
            "本次不会覆盖。确认没有下载进程后删除 .part，再运行 uv run download。",
        )
    opener = opener or urllib.request.build_opener(HTTPSRedirects()).open
    request = urllib.request.Request(
        _download_url(entry),
        headers={"User-Agent": "HandShake/0.2 public-data-downloader"},
    )
    emit(f"下载 {target.name}（最多 {limit // 1024**2} MiB；仅哈希通过后发布）…")
    written, digest, last_print = 0, hashlib.sha256(), time.monotonic()
    try:
        with _open_archive(request, opener) as response, partial.open("xb") as stream:
            content_type = response.headers.get("Content-Type", "").lower()
            length = response.headers.get("Content-Length")
            if "html" in content_type:
                raise WorkflowError(
                    "网盘返回了登录/确认页面，而不是数据包。",
                    f"请使用官方分享页手动下载到 {target}：{entry['url']}",
                )
            if length and (not length.isdecimal() or int(length) > limit):
                raise WorkflowError("下载响应大小不合法或超过预算。")
            while block := response.read(1024**2):
                written += len(block)
                if written > limit:
                    raise WorkflowError("数据包超过下载大小预算，已停止。")
                if written == len(block) and not block.startswith(b"PK"):
                    raise WorkflowError(
                        "下载内容不是 ZIP 数据包。", f"请检查官方分享页：{entry['url']}"
                    )
                digest.update(block)
                stream.write(block)
                if time.monotonic() - last_print >= 1:
                    emit(f"  已接收 {written / 1024**2:.1f} MiB")
                    last_print = time.monotonic()
            if not written or (length and written != int(length)):
                raise WorkflowError(
                    "下载不完整，未发布文件。", "网络恢复后重试 uv run download。"
                )
        if digest.hexdigest() != expected:
            rejected = target.with_suffix(f".rejected-{digest.hexdigest()[:12]}.zip")
            if rejected.exists():
                partial.unlink()
            else:
                partial.rename(rejected)
            raise WorkflowError(
                f"上游文件 SHA256 不一致：{target.name}\n  预期：{expected}\n  实际：{digest.hexdigest()}\n  隔离文件：{rejected}",
                "下载内容未作为正式数据发布。核对 data/SOURCES.md 的上游版本问题；不自动修改可信哈希。",
            )
        os.replace(partial, target)
        emit(f"校验通过：{target.name}（{written / 1024**2:.1f} MiB）")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise WorkflowError(
            f"下载失败：{error}",
            f"检查网络/代理后重试；或从 {entry['url']} 手动下载到 {target}。",
        ) from error
    finally:
        partial.unlink(missing_ok=True)


def download(ws, emit=print):
    entries = catalog(ws)
    directory = ws.path(ws.settings.data.raw)
    emit("来源：IPN HandS / CC BY 4.0。使用和报告时保留作者归属与论文引用。")
    with operation_lock(directory, "download"):
        for name, limit in LIMITS.items():
            fetch(entries[name], directory / name, limit=limit, emit=emit)
        write_json(
            directory / "download.json",
            {
                "format_version": "workflow-download-v1",
                "files": {name: entries[name]["expected_sha256"] for name in LIMITS},
            },
        )
    emit("数据包已就位。下一步：uv run prepare")
