"""Camera preview tests use fake/synthetic input; never open a physical camera."""

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.desktop import decode_command, serve
from backend.desktop_preview import CameraPreviewSession
from backend.perception.preview import CameraFrame, CameraPreviewWorker, encode_frame
from shared.preview_protocol import (
    MAGIC,
    MAX_JPEG_BYTES,
    PREFIX,
    PreviewCommand,
    PreviewPacket,
    PreviewSnapshot,
    read_packet,
)


class Clock:
    now = 10_000_000_000

    def __call__(self):
        return self.now


class Worker:
    def __init__(self):
        self.done = threading.Event()
        self.stop_requested = False
        self.started = False
        self.frame = None
        self.error = None

    def start(self):
        self.started = True

    def request_stop(self):
        self.stop_requested = True

    def take(self):
        frame, self.frame = self.frame, None
        return frame


def send(session, kind, index=1, epoch=None):
    session.accept(
        PreviewCommand(
            version=2,
            session_id=session.session_id,
            command_id=str(index),
            epoch=str(session.epoch if epoch is None else epoch),
            kind=kind,
        )
    )


def test_camera_is_opt_in_and_stop_waits_for_release():
    worker, clock = Worker(), Clock()
    session = CameraPreviewSession(worker_factory=lambda: worker, clock=clock)
    assert session.snapshot().snapshot.state == "stopped"
    assert not worker.started
    send(session, "start_camera")
    assert session.snapshot().snapshot.state == "starting"
    worker.frame = CameraFrame(
        8, clock.now, 160, 120, b"\xff\xd8\xff\xd9", (), "unavailable", 10.0
    )
    frame = session.snapshot()
    assert frame.snapshot.state == "preview"
    assert frame.snapshot.model_status == "unavailable"
    send(session, "stop", 2)
    assert worker.stop_requested
    stopping = session.snapshot()
    assert stopping.snapshot.state == "stopping"
    assert not stopping.jpeg and not stopping.snapshot.hands
    send(session, "start", 3)
    assert session.snapshot().snapshot.state == "stopping"
    worker.frame = CameraFrame(
        9, clock.now, 160, 120, b"\xff\xd8\xff\xd9", (), "unavailable", 10.0
    )
    worker.done.set()
    assert session.snapshot().snapshot.state == "stopped"
    send(session, "start", 4)
    assert session.snapshot().snapshot.source == "synthetic"


def test_camera_stale_frame_clears_image_and_requires_new_epoch():
    worker, clock = Worker(), Clock()
    session = CameraPreviewSession(worker_factory=lambda: worker, clock=clock)
    send(session, "start_camera")
    old_epoch = session.epoch
    worker.frame = CameraFrame(
        1,
        clock.now - 800_000_000,
        160,
        120,
        b"\xff\xd8\xff\xd9",
        (),
        "unavailable",
        10.0,
    )
    result = session.snapshot()
    assert result.snapshot.reason == "input_stalled"
    assert not result.jpeg
    worker.done.set()
    send(session, "start_camera", 2, old_epoch)
    assert session.snapshot().snapshot.state == "stopped"


def test_camera_startup_timeout_with_healthy_heartbeats():
    worker, clock = Worker(), Clock()
    session = CameraPreviewSession(worker_factory=lambda: worker, clock=clock)
    send(session, "start_camera")
    for index in range(2, 11):
        clock.now += 1_000_000_000
        send(session, "heartbeat", index)
        value = session.snapshot()
    assert value.snapshot.reason == "input_stalled"
    assert value.snapshot.state == "stopping"


def test_heartbeat_timeout_requests_release_not_fake_stop_ack():
    worker, clock = Worker(), Clock()
    session = CameraPreviewSession(worker_factory=lambda: worker, clock=clock)
    send(session, "start_camera")
    clock.now += 1_600_000_000
    assert session.snapshot().snapshot.state == "stopping"
    assert worker.stop_requested


@pytest.mark.parametrize(
    "error", ["perception_unavailable", "camera_unavailable", "preview_error"]
)
def test_worker_fault_never_falls_back_to_synthetic(error):
    worker = Worker()
    session = CameraPreviewSession(worker_factory=lambda: worker)
    send(session, "start_camera")
    worker.error = error
    worker.done.set()
    packet = session.snapshot()
    assert packet.snapshot.reason == error
    assert packet.snapshot.source == "camera"
    assert packet.snapshot.state == "stopped"
    assert not packet.jpeg


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 1},
        {"kind": "record"},
        {"kind": "enable_live"},
        {"epoch": "01"},
        {"camera": "/path"},
    ],
)
def test_command_v2_rejects_unsupported_capabilities(changes):
    body = {
        "version": 2,
        "session_id": "a" * 32,
        "command_id": "1",
        "epoch": "0",
        "kind": "start",
    } | changes
    with pytest.raises(ValidationError):
        decode_command(json.dumps(body).encode() + b"\n", PreviewCommand)


def test_binary_framing_roundtrip_fragmentation_and_truncation():
    packet = CameraPreviewSession().snapshot()
    encoded = packet.encode()

    class Fragmented(io.BytesIO):
        def read(self, size):
            return super().read(min(size, 3))

    assert read_packet(Fragmented(encoded)) == packet
    for cut in (1, 11, len(encoded) - 1):
        with pytest.raises(EOFError):
            read_packet(io.BytesIO(encoded[:cut]))
    for prefix in (
        PREFIX.pack(b"HSP9", 4, 0),
        PREFIX.pack(MAGIC, 20_000, 0),
        PREFIX.pack(MAGIC, 4, MAX_JPEG_BYTES + 1),
    ):
        with pytest.raises(ValueError):
            read_packet(io.BytesIO(prefix))


def test_inactive_packet_cannot_carry_camera_pixels():
    snapshot = CameraPreviewSession().snapshot().snapshot
    with pytest.raises(ValueError):
        PreviewPacket(snapshot, b"\xff\xd8\xff\xd9")
    with pytest.raises(ValidationError):
        PreviewSnapshot(**(snapshot.model_dump() | {"frame_width": 160}))


def test_output_handles_partial_writes():
    class ShortWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])

    output = ShortWriter()
    serve(
        io.BytesIO(),
        output,
        session=CameraPreviewSession(),
        decoder=lambda line: decode_command(line, PreviewCommand),
        encoder=lambda value: value.encode(),
    )
    stream = io.BytesIO(output.getvalue())
    snapshots = []
    while stream.tell() < len(output.getvalue()):
        snapshots.append(read_packet(stream).snapshot)
    assert snapshots[-1].state == "stopped"


def test_golden_schema_and_packet():
    schema = json.loads(Path("shared/preview_protocol.schema.json").read_text())
    assert schema == {
        "command": PreviewCommand.model_json_schema(),
        "snapshot": PreviewSnapshot.model_json_schema(),
    }
    packet = read_packet(
        io.BytesIO(Path("frontend/native/test/fixtures/preview_v2.bin").read_bytes())
    )
    assert packet.snapshot.frame_timestamp_ns == "9007199254740999"
    assert (
        packet.snapshot.source == "camera"
    )  # synthetic pixels exercise the camera wire branch
    assert (packet.snapshot.frame_width, packet.snapshot.frame_height) == (160, 120)


def test_encode_frame_keeps_rgb_and_geometry_identity():
    cv = pytest.importorskip("cv2")
    import numpy as np

    from backend.capture.camera import SyntheticSource
    from backend.perception.mediapipe_adapter import SyntheticDetector
    from backend.perception.pipeline import PerceptionPipeline

    source = SyntheticSource(frames=20, width=800, height=600)
    source.open()
    pipeline = PerceptionPipeline(SyntheticDetector())
    try:
        for _ in range(17):
            frame = source.read()
        observation = pipeline.process(frame)
        encoded = encode_frame(frame, observation, 10.0)
        assert (encoded.width, encoded.height) == (640, 480)
        assert encoded.timestamp_ns == frame.info.timestamp_ns
        assert len(encoded.hands) == 1
        assert (
            encoded.hands[0].landmarks[0].x
            == observation.observations[0].landmarks_image[0][0]
        )
        decoded = cv.imdecode(np.frombuffer(encoded.jpeg, np.uint8), cv.IMREAD_COLOR)
        assert abs(int(decoded[20, 20, 2]) - int(frame.rgb[20, 20, 0])) <= 3
        assert decoded[20, 20, 0] < 3
    finally:
        source.close()
        pipeline.close()


def test_real_worker_on_synthetic_input_closes_and_no_model_is_explicit():
    pytest.importorskip("cv2")
    from backend.capture.camera import SyntheticSource

    worker = CameraPreviewWorker(
        source_factory=lambda: SyntheticSource(frames=500, paced=True)
    )
    worker.start()
    deadline = time.monotonic() + 3
    frame = None
    while frame is None and time.monotonic() < deadline:
        frame = worker.take()
        time.sleep(0.01)
    worker.request_stop()
    assert worker.done.wait(2)
    assert frame is not None and frame.perception_status == "unavailable"
    assert worker.error is None


def test_close_failure_is_not_confirmed():
    pytest.importorskip("cv2")
    from backend.capture.camera import SyntheticSource

    class BadClose(SyntheticSource):
        def close(self):
            super().close()
            raise RuntimeError("release failed")

    worker = CameraPreviewWorker(source_factory=lambda: BadClose(frames=1))
    worker.start()
    worker.thread.join(3)
    assert not worker.thread.is_alive()
    assert not worker.done.is_set()
    assert worker.error == "preview_error"


def test_v2_subprocess_eof_without_camera():
    result = subprocess.run(
        [sys.executable, "-m", "backend.desktop_preview"],
        input=b"",
        capture_output=True,
        timeout=4,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stream = io.BytesIO(result.stdout)
    while stream.tell() < len(result.stdout):
        packet = read_packet(stream)
        assert packet.snapshot.source == "synthetic" and not packet.jpeg
