import numpy as np
import pytest

from backend.capture.camera import SyntheticSource
from backend.perception.mediapipe_adapter import RawHand
from backend.perception.pipeline import PerceptionPipeline
from backend.perception.quality import QualitySettings, geometry
from backend.perception.synthetic import skeleton
from shared.contracts import TransformMetadata
from shared.observations import CapturedFrame, FrameInfo


class SequenceDetector:
    metadata = {'kind':'test','synthetic':True}
    def __init__(self, results): self.results=iter(results); self.times=[]
    def detect(self, rgb, timestamp_ms):
        self.times.append(timestamp_ms)
        result=next(self.results)
        if isinstance(result,Exception): raise result
        return result
    def close(self): pass


def frame(index, timestamp=None, stream='s'):
    info = FrameInfo(stream,index,index*33_333_333 if timestamp is None else timestamp,160,120,TransformMetadata(),'synthetic')
    return CapturedFrame(info,np.zeros((120,160,3),np.uint8),0)


def pipeline(results):
    return PerceptionPipeline(SequenceDetector(results),now=lambda:1000)


def hand(x=.5): return RawHand(skeleton(x),'right',.8)


def test_no_hand_error_and_unprocessed_are_distinct():
    p=pipeline([[],RuntimeError('bad sdk'),[hand()]])
    assert p.process(frame(0)).status.perception_status.value=='processed_no_hand'
    error=p.process(frame(1))
    assert error.status.perception_status.value=='error' and 'bad sdk' in error.status.drop_or_error_reason
    assert p.process(frame(2)).observations[0].observation_valid
    assert p.process(frame(2)).status.perception_status.value=='not_processed'


def test_video_milliseconds_map_to_source_and_skip_same_ms():
    p=pipeline([[hand()],[hand()]])
    a=p.process(frame(0,10_000_000))
    b=p.process(frame(1,10_000_100))
    c=p.process(frame(2,11_000_000))
    assert a.status.source_timestamp_ns==10_000_000
    assert b.status.drop_or_error_reason=='same_mediapipe_millisecond'
    assert c.status.source_timestamp_ns==11_000_000
    assert p.detector.times==[0,1]
    with pytest.raises(ValueError,match='new stream'): p.process(frame(3,stream='new'))


def test_detector_array_order_does_not_define_track_identity():
    p=pipeline([[hand(.25),hand(.7)],[hand(.7),hand(.25)]])
    a=p.process(frame(0)).observations
    b=p.process(frame(1)).observations
    assert all(o.observation_valid for o in a+b)
    assert a[0].track_id==b[1].track_id and a[1].track_id==b[0].track_id


def test_crossing_invalidates_entire_frame_and_does_not_reuse_ids():
    p=pipeline([[hand(.3),hand(.7)],[hand(.49),hand(.51)],[hand(.3),hand(.7)]])
    a=p.process(frame(0)).observations
    b=p.process(frame(1)).observations
    c=p.process(frame(2)).observations
    assert all('association_ambiguous' in o.quality_flags and not o.observation_valid for o in b)
    assert {o.track_id for o in a}.isdisjoint({o.track_id for o in c})


def test_long_no_hand_gap_retires_track_id():
    p=pipeline([[hand()],[],[hand()]])
    a=p.process(frame(0)).observations[0]
    p.process(frame(1,1_000_000_000))
    b=p.process(frame(2,1_100_000_000)).observations[0]
    assert a.track_id!=b.track_id


@pytest.mark.parametrize('problem',['nan','small','edge','count','degenerate'])
def test_bad_geometry_mask_and_flags_preserved(problem):
    points=skeleton()
    expected={'nan':'nonfinite_landmarks','small':'palm_too_small','edge':'edge_out_of_bounds',
              'count':'landmark_count_or_shape','degenerate':'degenerate_bones'}[problem]
    if problem=='nan': points[8]=(float('nan'),.2,0.)
    if problem=='small': points=[(.5+(x-.5)*.001,.5+(y-.5)*.001,z*.001) for x,y,z in points]
    if problem=='edge': points[8]=(2.,.5,0.)
    if problem=='count': points=points[:4]
    if problem=='degenerate': points[8]=points[7]
    result=pipeline([[RawHand(points)]]).process(frame(0))
    obs=result.observations[0]
    assert not obs.observation_valid and expected in obs.quality_flags and obs.quality==0
    assert np.isfinite(obs.landmarks_image).all()
    if problem=='nan': assert not obs.validity_mask[8] and obs.handedness_score is None


def test_abnormal_motion_is_quality_failure_not_a_new_action():
    p=pipeline([[hand(.4)],[hand(.55)]])
    p.process(frame(0))
    obs=p.process(frame(1,1_000_000)).observations[0]
    assert 'abnormal_velocity' in obs.quality_flags


def test_world_estimates_are_optional_and_not_confidence():
    p=pipeline([[RawHand(skeleton(),'unknown',None,skeleton())]])
    result=p.process(frame(0))
    assert result.observations[0].handedness_score is None
    assert np.asarray(result.world_landmarks[0]).shape==(21,3)
    assert result.status.processed_at_ns==1000


def test_model_digest_checked_before_optional_sdk_import(tmp_path):
    from backend.perception.mediapipe_adapter import MediaPipeDetector
    asset=tmp_path/'fake.task'; asset.write_bytes(b'not-a-model')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        MediaPipeDetector(asset,expected_sha256='0'*64)


def test_sdk_adapter_video_options_and_optional_world_mapping(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from backend.perception import mediapipe_adapter as adapter
    asset=tmp_path/'fake.task'; asset.write_bytes(b'fixture')
    captured={}
    class Detector:
        def detect_for_video(self,image,ms):
            captured['ms']=ms
            return NS(hand_landmarks=[[NS(x=x,y=y,z=z) for x,y,z in skeleton()]],
                      handedness=[[NS(category_name='Right',score=.9)]],hand_world_landmarks=[])
        def close(self): captured['closed']=True
    def create(options): captured['options']=options; return Detector()
    fake=NS(tasks=NS(BaseOptions=lambda **kw:NS(**kw),vision=NS(HandLandmarkerOptions=lambda **kw:NS(**kw),
        RunningMode=NS(VIDEO='VIDEO'),HandLandmarker=NS(create_from_options=create))),
        Image=lambda **kw:NS(**kw),ImageFormat=NS(SRGB='RGB'))
    monkeypatch.setitem(sys.modules,'mediapipe',fake)
    monkeypatch.setattr(adapter,'version',lambda _:'fake-sdk')
    detector=adapter.MediaPipeDetector(asset)
    result=detector.detect(np.zeros((2,2,3),np.uint8),12)
    assert captured['options'].running_mode=='VIDEO' and captured['options'].num_hands==2
    assert result[0].handedness=='right' and result[0].world is None
    assert captured['ms']==12 and detector.metadata['sdk_version']=='fake-sdk'
    detector.close(); assert captured['closed']


def test_more_than_two_results_is_fault_not_partial_publication():
    result=pipeline([[hand(.2),hand(.5),hand(.8)]]).process(frame(0))
    assert result.status.perception_status.value=='error' and not result.observations
