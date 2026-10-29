"""Small procedural test skeletons; not a training dataset or model output."""
from backend.perception.mediapipe_adapter import RawHand


def skeleton(cx=.5, cy=.65):
    points = [(cx, cy, 0.)]
    for finger in range(5):
        for joint in range(4):
            points.append((cx + (finger - 2) * .035, cy - (joint + 1) * .035 - .015 * finger, -.002 * joint))
    return points


def synthetic_hands(index):
    phase = index % 90
    if phase < 15:
        return []
    if 60 <= phase < 70:
        points = skeleton()
        points[8] = (float("nan"), .4, 0.)
        return [RawHand(points)]
    if 70 <= phase < 80:
        return [RawHand(skeleton(.49)), RawHand(skeleton(.51))]
    return [RawHand(skeleton(.4 + (phase % 30) * .002), "right", .9)]
