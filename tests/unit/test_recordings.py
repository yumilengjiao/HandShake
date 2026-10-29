from pathlib import Path
import json
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest

from backend.capture.camera import SyntheticSource
from backend.collection.annotation import export_annotations
from backend.collection.recording_writer import AsyncRecorder, RecordingWriter
from backend.perception.mediapipe_adapter import SyntheticDetector
from backend.perception.pipeline import PerceptionPipeline
from shared.annotations import AnnotationInterval, AnnotationTimeline
from shared.contracts import Gesture
from shared.recordings import (atomic_json, digest_file, iter_recording, load_manifest,
                               recover_recording, load_chunk, save_npz)


def rows(count=85):
    source=SyntheticSource(frames=count,width=16,height=12)
    source.open()
    detector=PerceptionPipeline(SyntheticDetector())
    result=[]
    for _ in range(count):
        frame=source.read()
        result.append((detector.process(frame),frame.rgb))
    source.close()
    return result


def writer(tmp_path, **kwargs):
    return RecordingWriter(tmp_path,subject_id='synthetic-p1',session_id='synthetic-s1',
        consent_reference='synthetic-only',consent_keypoints=True,synthetic=True,
        source_timebase='synthetic',chunk_frames=8,**kwargs)


def test_no_consent_creates_no_files(tmp_path):
    with pytest.raises(PermissionError):
        RecordingWriter(tmp_path,subject_id='p1',session_id='s1',consent_reference='test')
    assert list(tmp_path.iterdir())==[]


def test_complete_timeline_round_trip_including_missing_and_invalid(tmp_path):
    w=writer(tmp_path)
    source=rows()
    w.event('start',0)
    for row,_ in source: w.append(row)
    w.event('prompt',30_000_000,target_gesture='wake_open',practice=True)
    w.event('pause',40_000_000)
    w.event('resume',50_000_000)
    w.event('end',source[-1][0].info.timestamp_ns)
    assert w.finish() and w.finish()
    restored=list(iter_recording(w.root))
    assert len(restored)==85
    assert [r.info.timestamp_ns for r in restored]==[r.info.timestamp_ns for r,_ in source]
    assert restored[0].status.perception_status.value=='processed_no_hand'
    assert not restored[65].observations[0].observation_valid
    assert restored[20].observations[0].landmarks_image==source[20][0].observations[0].landmarks_image
    manifest=load_manifest(w.root)
    assert manifest.complete and not manifest.context_available and manifest.synthetic
    assert not (w.root/'raw').exists()
    assert all(c.frames<=8 for c in manifest.chunks)
    assert not list(w.root.rglob('*.tmp'))
    for chunk in manifest.chunks:
        with np.load(w.root/chunk.path,allow_pickle=False) as data:
            assert all(not data[key].dtype.hasobject for key in data.files)
    events=[json.loads(line) for line in (w.root/'events.jsonl').read_text().splitlines()]
    assert [e['sequence'] for e in events]==list(range(5))
    assert events[1]['payload']['target_gesture']=='wake_open'


def test_raw_requires_separate_consent_and_has_exact_frame_mapping(tmp_path):
    row,rgb=rows(1)[0]
    w=writer(tmp_path)
    with pytest.raises(PermissionError): w.append(row,raw_rgb=rgb)
    w.finish(reason='test refusal')
    w=writer(tmp_path,consent_raw=True)
    w.append(row,raw_rgb=rgb)
    assert w.finish()
    manifest=load_manifest(w.root)
    entry=manifest.chunks[0]
    assert entry.raw_frames==1
    with np.load(w.root/entry.raw_path,allow_pickle=False) as data:
        assert data.files==['frame_000000']
        np.testing.assert_array_equal(data['frame_000000'],rgb)
    assert len(list(iter_recording(w.root)))==1


def test_cancelled_and_crash_remain_excluded_and_recovery_is_new_report(tmp_path):
    w=writer(tmp_path)
    for row,_ in rows(9): w.append(row)
    # One committed block, one uncommitted row simulates a lost process tail.
    w.events.close()
    with pytest.raises(ValueError,match='incomplete'): list(iter_recording(w.root))
    report=recover_recording(w.root)
    assert json.loads(report.read_text())['verified_committed_frames']==8
    assert not load_manifest(w.root,allow_incomplete=True).complete
    assert len(list(iter_recording(w.root,allow_incomplete=True)))==8
    cancelled=writer(tmp_path)
    for row,_ in rows(2): cancelled.append(row)
    assert not cancelled.finish(cancelled=True)
    assert load_manifest(cancelled.root,allow_incomplete=True).state=='cancelled'


def test_hash_and_path_escape_tampering_refused(tmp_path):
    w=writer(tmp_path)
    w.append(rows(1)[0][0]); w.finish()
    chunk=w.root/w.manifest.chunks[0].path
    with chunk.open('ab') as out: out.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'): list(iter_recording(w.root))
    raw=json.loads((w.root/'manifest.json').read_text())
    raw['chunks'][0]['path']='../../other.npz'
    atomic_json(w.root/'manifest.json',raw)
    with pytest.raises(ValueError): load_manifest(w.root)


def test_duplicate_or_switched_source_and_timebase_rejected(tmp_path):
    w=writer(tmp_path)
    row=rows(1)[0][0]
    w.append(row)
    with pytest.raises(ValueError,match='ordered'): w.append(row)
    different=rows(1)[0][0]
    with pytest.raises(ValueError,match='ordered'): w.append(different)
    w.finish(reason='intentional test')


def test_disk_failure_never_publishes_complete(tmp_path,monkeypatch):
    w=writer(tmp_path)
    def broken(*_): raise OSError('disk full')
    monkeypatch.setattr(w,'append',broken)
    recorder=AsyncRecorder(w)
    recorder.submit('append',rows(1)[0][0])
    deadline=monotonic()+1
    while not recorder.error and monotonic()<deadline: sleep(.001)
    assert not recorder.close()
    assert 'disk full' in recorder.error
    assert not load_manifest(w.root,allow_incomplete=True).complete


def test_bounded_disk_overflow_and_timeout_never_become_complete_later(tmp_path,monkeypatch):
    w=writer(tmp_path)
    gate,entered=Event(),Event()
    original=w.append
    def blocked(*args,**kwargs):
        entered.set(); gate.wait(2); original(*args,**kwargs)
    monkeypatch.setattr(w,'append',blocked)
    recorder=AsyncRecorder(w,capacity=1)
    source=rows(3)
    recorder.submit('append',source[0][0]); assert entered.wait(1)
    recorder.submit('append',source[1][0])
    with pytest.raises(RuntimeError,match='overflow'): recorder.submit('append',source[2][0])
    assert not recorder.close(timeout=.01)
    gate.set()
    # Wake the daemon's queue loop after timeout for deterministic test cleanup.
    recorder.queue.put(None,timeout=1)
    recorder.thread.join(1)
    assert not load_manifest(w.root,allow_incomplete=True).complete
    w.events.close()


def test_nonzero_source_origin_preserves_relative_frame_and_completion(tmp_path):
    from dataclasses import replace
    from shared.contracts import FrameStatus
    from shared.observations import ObservationFrame
    row,_=rows(1)[0]
    offset=8_000_000_000
    info=replace(row.info,timestamp_ns=offset)
    status=FrameStatus.model_validate(row.status.model_dump() | {
        'source_timestamp_ns':offset,'processed_at_ns':row.status.processed_at_ns+offset})
    shifted=ObservationFrame(info,status)
    w=writer(tmp_path,origin_ns=offset)
    w.append(shifted); w.event('start',offset); w.finish()
    loaded=list(iter_recording(w.root))[0]
    assert loaded.info.timestamp_ns==0
    assert loaded.status.processed_at_ns==row.status.processed_at_ns


def test_event_sequence_is_validated_even_with_updated_hash(tmp_path):
    w=writer(tmp_path)
    w.event('start',0); w.append(rows(1)[0][0]); w.finish()
    path=w.root/'events.jsonl'
    event=json.loads(path.read_text()); event['sequence']=7
    path.write_text(json.dumps(event)+'\n')
    manifest=w.manifest.model_dump(mode='json')
    manifest['events_sha256']=digest_file(path)
    atomic_json(w.root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='sequence'): list(iter_recording(w.root))


def test_annotations_uncertainty_export_revision_and_overlap_validation(tmp_path):
    w=writer(tmp_path)
    source=rows(3)
    for row,_ in source: w.append(row)
    w.finish()
    interval=AnnotationInterval(recording_id=w.manifest.recording_id,subject_id='synthetic-p1',session_id='synthetic-s1',
        hand_track_reference='manual-person-hand',annotator_id='a',start_ns=0,end_ns=10,
        gesture_label=None,class_valid=False,intent_label='ambiguous',intent_valid=False,phase='ambiguous',phase_valid=False)
    timeline=AnnotationTimeline(intervals=[interval])
    first=export_annotations(w.root,timeline)
    second=export_annotations(w.root,timeline)
    assert first!=second and first.read_text()==second.read_text()
    assert AnnotationTimeline.model_validate_json(first.read_text())==timeline
    with pytest.raises(ValueError,match='overlap'): AnnotationTimeline(intervals=[interval,interval])
    bad=interval.model_copy(update={'end_ns':10**15})
    with pytest.raises(ValueError,match='exceeds'): export_annotations(w.root,AnnotationTimeline(intervals=[bad]))
