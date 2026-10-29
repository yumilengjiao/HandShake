import json
from pathlib import Path
from time import sleep

import pytest

from backend.cli import main
from backend.capture.camera import SyntheticSource
from backend.collection.session import CollectionSession
from backend.perception.mediapipe_adapter import SyntheticDetector
from shared.recordings import iter_recording, load_manifest


def test_collect_cli_synthetic_then_inspect_no_real_actions(tmp_path,capsys):
    result=main(['collect','--source','synthetic','--frames','80','--fps','240','--width','32','--height','24',
                 '--headless','--consent-keypoints','--consent-reference','synthetic-only',
                 '--subject','synthetic-p','--session','synthetic-s','--output',str(tmp_path)])
    report=json.loads(capsys.readouterr().out)
    assert result==0 and report['complete'] and report['synthetic']
    assert report['control_backend'].startswith('none')
    recording=Path(report['recording_path'])
    manifest=load_manifest(recording)
    rows=list(iter_recording(recording))
    assert len(rows)==report['capture']['acquired']==80
    assert [r.info.frame_id for r in rows]==list(range(80))
    assert manifest.config_sha256!='0'*64
    assert main(['inspect','--recording',str(recording)])==0
    assert json.loads(capsys.readouterr().out)['verified_frames']==80
    assert not (recording/'raw').exists()


def test_no_model_and_no_consent_rejected_before_camera(tmp_path,capsys):
    assert main(['collect','--output',str(tmp_path/'absent')])==2
    assert 'consent' in capsys.readouterr().err and not (tmp_path/'absent').exists()
    assert main(['collect','--source','camera','--consent-keypoints','--consent-reference','test',
                 '--subject','p','--session','s','--output',str(tmp_path/'absent')])==2
    assert '--model' in capsys.readouterr().err and not (tmp_path/'absent').exists()


def test_preview_synthetic_does_not_create_dataset_or_change_default_dry_run(tmp_path,capsys):
    assert main(['preview','--source','synthetic','--frames','3','--fps','240','--width','16','--height','12',
                 '--headless','--output',str(tmp_path/'absent')])==0
    report=json.loads(capsys.readouterr().out)
    assert not report['complete'] and 'recording_path' not in report and not (tmp_path/'absent').exists()
    assert main([])==0
    assert 'ActionResult' in capsys.readouterr().out


@pytest.mark.parametrize('duration',['0','-1','nan','inf'])
def test_invalid_duration_rejected(duration,capsys):
    assert main(['preview','--headless','--duration',duration])==2
    assert 'finite and positive' in capsys.readouterr().err


def test_session_pause_prompt_retry_cancel_preserves_timeline(tmp_path):
    session=CollectionSession(SyntheticSource(frames=100,fps=120,width=16,height=12,paced=True),SyntheticDetector(),
        recording=dict(root=tmp_path,subject_id='synthetic-p',session_id='synthetic-s',consent_reference='synthetic-only',consent_keypoints=True))
    session.start()
    for _ in range(200):
        if session.recorder: break
        sleep(.005)
    assert session.recorder
    session.next_prompt(practice=True)
    session.toggle_pause()
    sleep(.03)
    session.toggle_pause()
    session.mark('retry',target_gesture=session.prompt)
    session.request_stop(cancelled=True)
    assert session.stop()
    manifest=load_manifest(session.recording_path,allow_incomplete=True)
    assert manifest.state=='cancelled' and not manifest.complete
    rows=list(iter_recording(session.recording_path,allow_incomplete=True))
    assert len(rows)>0
    events=[json.loads(s) for s in (session.recording_path/'events.jsonl').read_text().splitlines()]
    assert {'start','pause','resume','prompt','retry','cancel'}<={e['kind'] for e in events}
