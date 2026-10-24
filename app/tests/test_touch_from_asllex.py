import numpy as np

from training.touch_from_asllex import annotate, weights

# MediaPipe-like world pose (metres, y down): shoulders 0.4 m apart at y 0, hips 0.5 m below, face above
POSE = np.zeros((33, 3))
POSE[11], POSE[12], POSE[23], POSE[24] = [0.2, 0, 0], [-0.2, 0, 0], [0.12, 0.5, 0], [-0.12, 0.5, 0]
POSE[0], POSE[2], POSE[5], POSE[9], POSE[10] = [0, -0.3, -0.1], [0.03, -0.33, -0.1], [-0.03, -0.33, -0.1], [0.02, -0.27, -0.1], [-0.02, -0.27, -0.1]
HAND = np.zeros((21, 3))  # a point hand: every landmark at the wrist


def frame(wrist, left_wrist=None) -> dict:
    p = POSE.copy()
    p[16] = wrist
    f = {"pose": p.tolist(), "right": HAND.tolist(), "rightW": HAND.tolist(), "left": None, "leftW": None}
    if left_wrist is not None:
        p[15] = left_wrist
        f.update(pose=p.tolist(), left=HAND.tolist(), leftW=HAND.tolist())
    return f


def test_a_hold_counts_and_a_pass_does_not():
    hold = np.array([0.8] * 5 + [0.1] * 12 + [0.8] * 5)
    w = weights(hold, near=0.35, fps=30)
    assert (w[6:16] == 1).all() and w[0] == 0 and w[-1] == 0
    passing = np.array([0.8] * 5 + [0.6, 0.3, 0.1, 0.3, 0.6] + [0.8] * 5)  # through the place in 5 frames (0.17 s)
    assert not weights(passing, near=0.35, fps=30).any()


def test_a_hold_short_of_the_estimated_place_still_counts():
    w = weights(np.array([0.9] * 5 + [0.45] * 12 + [0.9] * 5), near=0.25, fps=30)  # FATHER: closest hold 0.45 away
    assert (w[6:16] == 1).all() and w[0] == 0
    assert not weights(np.array([0.9] * 5 + [0.7] * 12 + [0.9] * 5), near=0.25, fps=30).any()  # never near: left alone


def test_untracked_frames_are_not_contacts():
    assert not weights(np.full(10, np.inf), near=0.35, fps=30).any()


def test_chest_contact_ignores_depth():
    # MY: the hand comes in to the upper chest and rests there; MediaPipe puts it 30 cm in front, which must not matter
    path = [[-0.3, 0.45, -0.3]] * 3 + [[0, 0.14, -0.3]] * 12 + [[-0.3, 0.45, -0.3]] * 3
    clip = {"fps": 30, "frames": [frame(w) for w in path]}
    outcome, _ = annotate(clip, {"Contact.2.0": 1, "MajorLocation.2.0": "Body", "MinorLocation.2.0": "TorsoTop"})
    assert outcome == "touch"
    w = [f.get("touchR", 0) for f in clip["frames"]]
    assert w[9] == 1 and w[0] == 0


def test_hand_contact_marks_the_meeting_frames():
    apart, together = frame([-0.3, 0.2, -0.2], [0.1, 0.2, -0.3]), frame([0.1, 0.2, -0.1], [0.1, 0.2, -0.3])
    clip = {"fps": 30, "frames": [apart] * 4 + [together] * 10}
    outcome, _ = annotate(clip, {"Contact.2.0": 1, "MajorLocation.2.0": "Hand", "MinorLocation.2.0": "Palm"})
    assert outcome == "meetR"
    assert clip["frames"][9]["meetR"] == 1 and "meetR" not in clip["frames"][0]


def test_signs_without_contact_get_no_weights_and_old_ones_are_cleared():
    clip = {"fps": 30, "frames": [dict(frame([0, 0.2, -0.3]), touchR=1.0)] * 2}
    assert annotate(clip, {"Contact.2.0": 0})[0] == "no contact"
    assert all("touchR" not in f for f in clip["frames"])


# --- image-detected face contact (clips ASL-LEX gives no contact) ---
def image_frame(hand_xy) -> dict:
    """A 16:9 image of POSE (x' = .5 + x/1.6, y' = .5 + y/.9) with the point hand's 21 landmarks all at image point hand_xy."""
    f = frame([0, 0.2, -0.3])
    f["pose2d"] = [[0.5 + x / 1.6, 0.5 + y / 0.9] for x, y, _ in POSE]
    f["right"] = [list(hand_xy)] * 21
    return f


def chin_xy():
    m = (np.array(POSE[9]) + POSE[10]) / 2
    chin = m + (m - POSE[0]) * 0.35
    return [0.5 + chin[0] / 1.6, 0.5 + chin[1] / 0.9]


def run(path):
    clip = {"fps": 30, "frames": [image_frame(p) for p in path]}
    return annotate(clip, {})[0], clip["frames"]


def test_image_chin_hold_gets_touch_and_a_chin_place():
    far, chin = [0.5 + 0.3 / 1.6, 0.9], chin_xy()
    outcome, fr = run([far] * 3 + [chin] * 12 + [far] * 3)
    assert outcome == "image"
    w = [f.get("touchR", 0) for f in fr]
    assert w[0] == 0 and w[9] == 1 and 0 < w[3] < 1 and w[-1] == 0
    at = np.array(fr[9]["touchAtR"])
    m = (np.array(POSE[9]) + POSE[10]) / 2
    assert np.allclose(at[:2], (m + (m - POSE[0]) * 0.35)[:2], atol=1e-3) and at[2] == POSE[0][2]


def test_image_two_frame_pass_and_far_hand_get_nothing():
    far, chin = [0.5 + 0.3 / 1.6, 0.9], chin_xy()
    assert run([far] * 4 + [chin] * 2 + [far] * 4)[0] == "no contact"
    assert run([far] * 4 + [chin] * 6 + [far] * 4)[0] == "no contact"  # shorter than the ASL-LEX dwell (9 frames at 30 fps)
    # 0.3 shoulder widths (0.12 m) beside the chin
    assert run([[chin_xy()[0] + 0.3 * 0.4 / 1.6, chin_xy()[1]]] * 10)[0] == "no contact"


def test_a_held_run_has_one_touch_place_despite_a_jittering_nose():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(14):
        f = image_frame(chin_xy())
        for k in ("pose", "pose2d"):
            f[k] = [list(v) for v in f[k]]
        f["pose"][0][0] += rng.normal(0, 0.004)  # nose jitter in world x and y
        f["pose"][0][1] += rng.normal(0, 0.004)
        frames.append(f)
    annotate({"fps": 30, "frames": frames}, {})
    places = {tuple(f["touchAtR"]) for f in frames if "touchAtR" in f}
    assert len(places) == 1 and sum("touchAtR" in f for f in frames) >= 10


def test_image_contact_only_for_head_signs_or_signs_missing_from_asllex():
    far, chin = [0.5 + 0.3 / 1.6, 0.9], chin_xy()
    path = [far] * 3 + [chin] * 12 + [far] * 3
    outcomes = {}
    for name, props in {"neutral": {"MajorLocation.2.0": "Neutral"}, "head": {"MajorLocation.2.0": "Head"}, "missing": {}}.items():
        clip = {"fps": 30, "frames": [image_frame(p) for p in path]}
        outcomes[name] = annotate(clip, props)[0]
    assert outcomes == {"neutral": "no contact", "head": "image", "missing": "image"}
