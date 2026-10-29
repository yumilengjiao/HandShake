"""Offline skeleton timeline and versioned manual annotation import/export.

Prompts are shown as context only: no model prediction or prompt becomes a label.
"""
from pathlib import Path
from time import monotonic
from uuid import uuid4

from shared.annotations import AnnotationInterval, AnnotationTimeline
from shared.contracts import Gesture
from shared.recordings import atomic_json, iter_recording, load_manifest, read_events


def export_annotations(root, timeline):
    manifest = load_manifest(root, allow_incomplete=True)
    timeline = AnnotationTimeline.model_validate(timeline)
    rows = list(iter_recording(root, allow_incomplete=True))
    if not rows:
        raise ValueError("cannot annotate an empty recording")
    end = rows[-1].info.timestamp_ns + 1
    for interval in timeline.intervals:
        if (interval.recording_id, interval.subject_id, interval.session_id) != (
                manifest.recording_id, manifest.subject_id, manifest.session_id):
            raise ValueError("annotation recording/subject/session mismatch")
        if interval.end_ns > end:
            raise ValueError("annotation exceeds recorded timeline")
    target = Path(root) / "annotations" / f"{uuid4().hex}.json"
    target.parent.mkdir(exist_ok=True)
    atomic_json(target, timeline.model_dump(mode="json"))
    return target


def annotation_view(root, *, annotator_id, hand_reference="manual-hand-1", existing=None):
    from backend.capture.camera import _cv2
    from backend.collection.views import render_observations

    manifest = load_manifest(root, allow_incomplete=True)
    rows = list(iter_recording(root, allow_incomplete=True))
    if not rows:
        raise ValueError("no committed frames")
    cv = _cv2()
    events = read_events(root, allow_incomplete=True)
    title = "HandShake skeleton annotation (manual truth)"
    timeline = AnnotationTimeline.model_validate_json(Path(existing).read_text()) if existing else AnnotationTimeline(intervals=[])
    intervals = list(timeline.intervals)
    position, begin, end = 0, None, None
    labels = [None] + list(Gesture)
    label_index = 0
    intents = ["ambiguous", 0, 1]
    phases = ["ambiguous", "background", "preparation", "active", "release"]
    intent_index = phase_index = 0
    playing = False
    last_tick = monotonic()
    message = "labels default uncertain; prompts are NOT truth"
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, 1000, 700)
    cv.createTrackbar("frame", title, 0, max(1, len(rows)-1), lambda _: None)
    try:
        while True:
            position = min(cv.getTrackbarPos("frame", title), len(rows)-1)
            row = rows[position]
            image = render_observations(row)
            # Panel avoids drawing text outside small input images.
            panel = cv.copyMakeBorder(image, 0, 160, 0, max(0, 1000-image.shape[1]), cv.BORDER_CONSTANT)
            hints = [e for e in events if e.timestamp_ns <= row.info.timestamp_ns]
            hint = max(hints, key=lambda e:(e.timestamp_ns,e.sequence)) if hints else None
            hint_text = f"context ONLY: {hint.kind} {hint.payload}" if hint else "no prompt/context"
            texts = ["a/d or slider: step | SPACE play | b/e start/end | g label | i intent | p phase",
                     "ENTER add interval | u undo | s export new revision | q quit",
                     f"label={labels[label_index]} intent={intents[intent_index]} phase={phases[phase_index]}",
                     f"[start,end) ns: {begin},{end} intervals={len(intervals)} complete={manifest.complete}", message, hint_text]
            for j, text in enumerate(texts):
                cv.putText(panel, text, (10, image.shape[0]+25+25*j), cv.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            cv.imshow(title, panel)
            key = cv.waitKey(15) & 0xff
            if key in (27, ord('q')) or cv.getWindowProperty(title, cv.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                playing = not playing
                last_tick = monotonic()
            elif key in (ord('a'), ord('d')):
                position = max(0, min(len(rows)-1, position + (-1 if key == ord('a') else 1)))
                cv.setTrackbarPos("frame", title, position)
                playing = False
            elif key == ord('b'):
                begin = row.info.timestamp_ns
            elif key == ord('e'):
                end = row.info.timestamp_ns + 1
            elif key == ord('g'):
                label_index = (label_index + 1) % len(labels)
            elif key == ord('i'):
                intent_index = (intent_index + 1) % len(intents)
            elif key == ord('p'):
                phase_index = (phase_index + 1) % len(phases)
            elif key == ord('u') and intervals:
                intervals.pop()
                message = "removed last unsaved interval (prior exports unchanged)"
            elif key in (10, 13):
                try:
                    interval = AnnotationInterval(recording_id=manifest.recording_id, subject_id=manifest.subject_id,
                        session_id=manifest.session_id, hand_track_reference=hand_reference, annotator_id=annotator_id,
                        start_ns=begin, end_ns=end, gesture_label=labels[label_index], class_valid=labels[label_index] is not None,
                        intent_label=intents[intent_index], intent_valid=intents[intent_index] != "ambiguous",
                        phase=phases[phase_index], phase_valid=phases[phase_index] != "ambiguous")
                    AnnotationTimeline(intervals=intervals + [interval])
                    intervals.append(interval)
                    message = "interval added; edit detailed reference/target fields via validated JSON import"
                except (ValueError, TypeError) as error:
                    message = str(error).splitlines()[0]
            elif key == ord('s'):
                target = export_annotations(root, AnnotationTimeline(intervals=intervals))
                message = f"saved revision {target.name}"
            if playing and position < len(rows)-1:
                gap = (rows[position+1].info.timestamp_ns - row.info.timestamp_ns) / 1e9
                if monotonic() - last_tick >= gap:
                    cv.setTrackbarPos("frame", title, position+1)
                    last_tick = monotonic()
    finally:
        cv.destroyWindow(title)
