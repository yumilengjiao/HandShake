from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest

from backend.capture.camera import CaptureError, OpenCVSource, SyntheticSource
from backend.capture.clock import StreamClock
from backend.capture.worker import CaptureWorker
from shared.observations import CapturedFrame


def wait_until(predicate):
    deadline = monotonic() + 2
    while not predicate() and monotonic() < deadline:
        sleep(.001)
    assert predicate()


@pytest.mark.parametrize('times', [(5,5), (5,4), (5,-1), (5,True), (5,1.5)])
def test_clock_rejects_invalid_timestamps_without_relabeling(times):
    clock = StreamClock()
    clock.stamp(times[0])
    with pytest.raises(ValueError):
        clock.stamp(times[1])
    assert clock.last_ns == 5 and clock.frame_id == 0
    previous = clock.stream_id
    clock.reset()
    assert clock.stamp(0)[0] != previous


def test_array_boundary_does_not_share_mutable_source_pixels():
    source = SyntheticSource(frames=2, width=3, height=2)
    source.open()
    frame = source.read()
    original = np.ones((2,3,3), np.uint8)
    copied = CapturedFrame(frame.info, original, 0)
    original[:] = 0
    assert copied.rgb.sum() == 18
    with pytest.raises(ValueError):
        copied.rgb[0,0,0] = 1
    packet = copied.to_packet()
    assert packet.image_size == [3,2] and packet.pixel_format == 'RGB8'
    assert packet.image[0][0] == [1,1,1]
    source.close()


def test_latest_only_keeps_drop_statuses_not_images():
    worker = CaptureWorker(SyntheticSource(frames=30, width=2, height=2))
    worker.start()
    wait_until(lambda: worker.done)
    skipped, frame = worker.take(0)
    assert frame.info.frame_id == 29
    assert len(skipped) == 29
    assert all(row.status.perception_status.value == 'not_processed' for row in skipped)
    assert [r.info.frame_id for r in skipped] == list(range(29))
    assert all(not hasattr(row, 'rgb') for row in skipped)
    assert worker.stats.max_pending_frames == 1
    assert worker.stop() and worker.stop()


def test_status_ledger_overflow_is_visible_not_no_hand():
    worker = CaptureWorker(SyntheticSource(frames=20, width=2, height=2), status_capacity=3)
    worker.start()
    wait_until(lambda: worker.done)
    skipped, _ = worker.take(0)
    assert len(skipped) == 3 and worker.stats.status_overflow == 16
    worker.stop()


def test_native_read_hang_has_bounded_stop_and_reports_failure():
    gate, entered = Event(), Event()
    class BlockedSource:
        metadata = {}
        def open(self): pass
        def read(self):
            entered.set()
            gate.wait(2)
            raise StopIteration
        def close(self): pass
    worker = CaptureWorker(BlockedSource())
    worker.start()
    assert entered.wait(1)
    assert not worker.stop(timeout=.01)
    assert 'deadline' in worker.error
    gate.set()
    assert worker.stop(timeout=1)


def test_file_backpressure_consumes_all_frames_without_latest_skips():
    worker = CaptureWorker(SyntheticSource(frames=12,width=2,height=2),backpressure=True)
    worker.start()
    seen=[]
    while True:
        skipped, item = worker.take(.1)
        assert not skipped
        if item: seen.append(item.info.frame_id)
        if worker.done and item is None: break
        sleep(.001)
    assert seen==list(range(12)) and worker.stats.consumer_skipped==0
    assert worker.stop()


def test_paced_synthetic_stop_interrupts_long_wait():
    worker=CaptureWorker(SyntheticSource(frames=5,fps=.1,width=2,height=2,paced=True))
    worker.start()
    worker.take(.1)
    assert worker.stop(timeout=.1)


class FakeCapture:
    def __init__(self, frames, *, opened=True, fps=30.):
        self.frames = iter(frames)
        self.opened, self.fps = opened, fps
        self.released = False
    def isOpened(self): return self.opened
    def getBackendName(self): return 'test-backend'
    def set(self, prop, value): return False
    def get(self, prop): return {1:2,2:1,3:self.fps,5:0.}.get(prop,0.)
    def read(self): return next(self.frames, (False,None))
    def release(self): self.released=True


class FakeCV:
    __version__ = 'test'
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_BUFFERSIZE, CAP_PROP_POS_MSEC = range(1,6)
    COLOR_BGR2RGB = 1
    def __init__(self, cap): self.cap = cap
    def VideoCapture(self, *_): return self.cap
    def cvtColor(self, array, _): return array[:,:,::-1].copy()


def test_opencv_color_rotation_metadata_clock_and_reconnect():
    pixels = np.array([[[1,2,3],[4,5,6]]],np.uint8)
    times = iter([100,200])
    cap = FakeCapture([(True,pixels),(True,pixels)])
    source = OpenCVSource(cv=FakeCV(cap), now=lambda:next(times), rotation=90)
    source.open()
    first = source.read()
    assert first.info.width == 1 and first.info.height == 2
    assert first.rgb[:,0].tolist() == [[3,2,1],[6,5,4]]
    assert first.info.timestamp_ns == 100
    assert not source.metadata['requests']['buffer_size']['set_accepted']
    assert source.metadata['exposure_age_known'] is False
    source.close()
    assert cap.released
    source.open()
    second = source.read()
    assert first.info.stream_id != second.info.stream_id and second.info.frame_id == 0
    source.close()


def test_camera_failure_is_not_file_eof_and_releases_on_open_error():
    cap = FakeCapture([], opened=False)
    source = OpenCVSource(cv=FakeCV(cap))
    with pytest.raises(CaptureError): source.open()
    assert cap.released
    cap = FakeCapture([])
    source = OpenCVSource(cv=FakeCV(cap))
    source.open()
    with pytest.raises(CaptureError, match='read failed'): source.read()
    source.close()


def test_file_uses_media_not_wall_time_and_duplicate_decoder_time_rejected(tmp_path):
    path = tmp_path / 'test.avi'
    path.touch()
    pixels = np.zeros((1,2,3),np.uint8)
    cap = FakeCapture([(True,pixels)]*3, fps=25.)
    source = OpenCVSource(video=path, cv=FakeCV(cap))
    source.open()
    assert [source.read().info.timestamp_ns for _ in range(3)] == [0,40_000_000,80_000_000]
    with pytest.raises(StopIteration): source.read()
    source.close()
    cap = FakeCapture([(True,pixels)]*2)
    source = OpenCVSource(video=path, cv=FakeCV(cap), video_clock='decoder')
    source.open()
    source.read()
    with pytest.raises(ValueError, match='duplicate'): source.read()
    source.close()
