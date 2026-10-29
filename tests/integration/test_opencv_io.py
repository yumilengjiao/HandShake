"""Optional real OpenCV IO tests use generated pixels only, never a camera."""
import numpy as np
import pytest

cv = pytest.importorskip('cv2', reason='optional perception extra not installed')

from backend.capture.camera import OpenCVSource
from backend.capture.worker import CaptureWorker
from backend.collection.views import render_observations
from backend.perception.pipeline import PerceptionPipeline
from backend.perception.mediapipe_adapter import SyntheticDetector


def test_real_generated_video_decoding_preserves_all_frames_and_source_clock(tmp_path):
    path=tmp_path/'synthetic.avi'
    writer=cv.VideoWriter(str(path),cv.VideoWriter_fourcc(*'MJPG'),25.,(64,48))
    assert writer.isOpened()
    try:
        for i in range(8): writer.write(np.full((48,64,3),i*20,np.uint8))
    finally:
        writer.release()
    source=OpenCVSource(video=path)
    worker=CaptureWorker(source,backpressure=True)
    worker.start()
    frames=[]
    try:
        while True:
            skipped,frame=worker.take(.2)
            assert not skipped
            if frame: frames.append(frame)
            if worker.done and frame is None: break
        assert not worker.error
        assert [f.info.timestamp_ns for f in frames]==[i*40_000_000 for i in range(8)]
        assert all(f.rgb.shape==(48,64,3) for f in frames)
        assert source.metadata['kind']=='file'
    finally:
        assert worker.stop()


def test_real_opencv_overlay_uses_copy_and_handles_invalid_skeleton():
    from backend.capture.camera import SyntheticSource
    source=SyntheticSource(frames=70,width=160,height=120)
    source.open()
    p=PerceptionPipeline(SyntheticDetector())
    try:
        for _ in range(66): frame=source.read()
        row=p.process(frame)
        assert not row.observations[0].observation_valid
        before=frame.rgb.copy()
        image=render_observations(row,frame.rgb,mirror=True)
        assert image.shape==frame.rgb.shape and image.dtype==np.uint8
        np.testing.assert_array_equal(before,frame.rgb)
    finally:
        source.close(); p.close()
