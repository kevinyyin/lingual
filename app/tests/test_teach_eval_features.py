import math

from server.protocol import Frame
from training.teach_eval.features import features

META = {"w": 640, "h": 480, "fps": 15.0}


def clip(hand_at=(0.45, 0.55), gap=0.02, yaw=lambda i: 0.0, n=20, labels_swapped=False):
    """A signer (aspect 4:3) holding the right hand at `hand_at`, fingers `gap` apart, head yaw per frame."""
    frames = []
    for i in range(n):
        pose = [[0.5, 0.6, 0.0, 1.0] for _ in range(33)]
        pose[0] = [0.5, 0.35, 0.0, 1.0]
        pose[7], pose[8] = [0.55, 0.34, 0.0, 1.0], [0.45, 0.34, 0.0, 1.0]
        pose[9], pose[10] = [0.51, 0.39, 0.0, 1.0], [0.49, 0.39, 0.0, 1.0]
        pose[11], pose[12] = [0.6, 0.6, 0.0, 1.0], [0.4, 0.6, 0.0, 1.0]
        pose[23], pose[24] = [0.55, 0.95, 0.0, 1.0], [0.45, 0.95, 0.0, 1.0]
        x, y = hand_at[0] + 0.002 * i, hand_at[1]
        pose[16] = [x, y, 0.0, 1.0]  # signer's right wrist
        pose[15] = [0.6, 1.0, 0.0, 1.0]  # left hand down at rest
        hand = [[x, y, 0.0]] + [[0.0, 0.0, 0.0]] * 20
        for f_i, base in enumerate((1, 5, 9, 13, 17)):  # thumb..little: knuckles fixed, fingers fanned `gap` apart
            for j in range(4):
                hand[base + j] = [x + (f_i - 2) * (0.012 + gap * j), y - 0.02 - 0.015 * j, 0.0]
        right = hand
        frames.append(Frame(t=i * 1000 / 15, pose=pose, head=(yaw(i), 0.0, 0.0),
                            left=right if labels_swapped else None, right=None if labels_swapped else right))
    return frames


def test_a_hand_at_the_face_reads_as_near_the_face():
    near = features(clip(hand_at=(0.5, 0.42)), META)
    low = features(clip(hand_at=(0.45, 0.8)), META)
    assert near["near_face_frac"] == 1.0 and low["near_face_frac"] == 0.0
    assert near["wrist_nose_min"] < low["wrist_nose_min"]
    assert near["loc_bin"] in ("head", "neck") and low["loc_bin"] in ("torso", "low")


def test_fingers_close_together_have_small_gaps():
    tight, spread = features(clip(gap=0.004), META), features(clip(gap=0.03), META)
    assert tight["finger_gap_mean"] < 0.5 * spread["finger_gap_mean"]


def test_a_turning_head_shows_yaw_range_and_speed():
    still = features(clip(), META)
    turning = features(clip(yaw=lambda i: 25 * math.sin(i / 3)), META)
    assert still["head_yaw_range"] == 0 and turning["head_yaw_range"] > 40 and turning["head_speed_max"] > 50


def test_hand_labels_are_checked_against_the_pose_wrists():
    ok, swapped = features(clip(), META), features(clip(labels_swapped=True), META)
    # an unmirrored camera: MediaPipe calls the signer's right hand "Left"; our clip labels it "right" (a flip)
    assert ok["handedness_flip_frac"] == 1.0 and swapped["handedness_flip_frac"] == 0.0
    assert ok["left_handed"] == 0 and ok["hand_detect_frac"] == 1.0
