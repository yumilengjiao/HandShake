"""Local, explicitly supplied MediaPipe Hand Landmarker asset, VIDEO mode only."""
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path


@dataclass(frozen=True)
class RawHand:
    landmarks: object
    handedness: str = "unknown"
    score: float | None = None
    world: object = None


class MediaPipeDetector:
    def __init__(self, model_path, *, expected_sha256=None, source="user-supplied-local-asset", license_note="verify asset license before redistribution", max_hands=2):
        if max_hands not in (1, 2):
            raise ValueError("max_hands must be 1 or 2")
        path = Path(model_path)
        digest = sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError("MediaPipe asset SHA256 mismatch")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("MediaPipe unavailable; install the perception extra") from error
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=max_hands)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.metadata = {"kind": "mediapipe_hand_landmarker", "sdk_version": version("mediapipe"),
                         "asset_sha256": digest, "asset_source": source, "asset_license": license_note,
                         "mode": "VIDEO", "max_hands": max_hands, "synthetic": False}

    def detect(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            categories = result.handedness[i] if i < len(result.handedness) else []
            name = categories[0].category_name.lower() if categories else "unknown"
            score = float(categories[0].score) if categories else None
            world = None
            if i < len(result.hand_world_landmarks):
                world = [(p.x,p.y,p.z) for p in result.hand_world_landmarks[i]]
            hands.append(RawHand([(p.x,p.y,p.z) for p in landmarks], name, score, world))
        return hands

    def close(self):
        self.detector.close()


class SyntheticDetector:
    """Explicit test data, NEVER used as a fallback for missing MediaPipe assets."""
    metadata = {"kind": "synthetic_landmarks", "synthetic": True, "mode": "fixture"}

    def detect(self, rgb, timestamp_ms):
        from backend.perception.synthetic import synthetic_hands
        return synthetic_hands(int(rgb[0,0,0]))

    def close(self):
        pass
