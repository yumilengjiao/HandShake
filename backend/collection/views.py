"""Small OpenCV tool views, NOT the T20 product UI/framework decision."""
import numpy as np

from backend.capture.camera import _cv2
from backend.perception.quality import BONES


def render_observations(row, rgb=None, *, mirror=False):
    cv = _cv2()
    image = np.zeros((row.info.height, row.info.width, 3), dtype=np.uint8) if rgb is None else np.array(rgb, copy=True)
    for obs in row.observations:
        color = (30, 220, 80) if obs.observation_valid else (240, 70, 30)
        points = [(int(np.clip(p[0], -2, 3) * row.info.width), int(np.clip(p[1], -2, 3) * row.info.height)) for p in obs.landmarks_image]
        for a, b in BONES:
            if obs.validity_mask[a] and obs.validity_mask[b]:
                cv.line(image, points[a], points[b], color, 2)
        for p, valid in zip(points, obs.validity_mask):
            if valid:
                cv.circle(image, p, 3, color, -1)
    if mirror:
        image = image[:, ::-1].copy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.putText(image, f"frame={row.info.frame_id} t={row.info.timestamp_ns/1e9:.3f}s {row.status.perception_status.value}",
               (8, 18), cv.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    for i, obs in enumerate(row.observations):
        cv.putText(image, f"{obs.track_id} {obs.handedness} q={obs.quality:.1f} {','.join(obs.quality_flags)}",
                   (8, 36+16*i), cv.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    return image


class PreviewWindow:
    def __init__(self, title="HandShake collection (NO DEVICE CONTROL)"):
        self.cv = _cv2()
        self.title = title
        self.cv.namedWindow(title, self.cv.WINDOW_NORMAL)
        self.cv.resizeWindow(title, 800, 600)

    def show(self, session):
        snapshot = session.snapshot()
        if snapshot:
            frame, row = snapshot
            image = render_observations(row, frame.rgb, mirror=frame.info.transform.preview_mirrored)
            self.cv.putText(image, f"{session.prompt} {'PAUSED (still recording)' if session.paused else ''}",
                            (8, image.shape[0]-12), self.cv.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            self.cv.imshow(self.title, image)
        key = self.cv.waitKey(1) & 0xff
        if key in (27, ord('q')) or self.cv.getWindowProperty(self.title, self.cv.WND_PROP_VISIBLE) < 1:
            session.request_stop()
        elif key == ord('x'):
            session.request_stop(cancelled=True)
        elif key == ord('p'):
            session.toggle_pause()
        elif key in (ord('n'), ord('r')):
            if key == ord('r'):
                session.mark("retry", target_gesture=session.prompt)
            session.next_prompt(practice=key == ord('r'))

    def close(self):
        self.cv.destroyWindow(self.title)
