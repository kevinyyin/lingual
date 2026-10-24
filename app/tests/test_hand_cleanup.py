import numpy as np

from training.hand_cleanup import FINGER_BONES, clean_track, fix_knuckles, palm_frame, segments

PALM = 0.09
BASES = {1: (0.02, 0.03), 5: (0.09, 0.025), 9: (0.09, 0.0), 13: (0.085, -0.02), 17: (0.075, -0.04)}


def hand(curl_deg: float = 0.0, rot: np.ndarray | None = None) -> np.ndarray:
    """A synthetic right hand in metres: palm in the x-y plane (x = wrist -> middle knuckle), each finger bent by
    curl_deg at every joint toward -z (a fist at ~90)."""
    h = np.zeros((21, 3))
    for chain, ratios in FINGER_BONES:
        h[chain[0]] = [*BASES[chain[0]], 0.0]
        a = 0.0
        for (i, j), r in zip(zip(chain, chain[1:]), ratios):
            a += np.radians(curl_deg) if chain[0] != 1 else 0.0
            h[j] = h[i] + np.array([np.cos(a), 0.0, -np.sin(a)]) * r * PALM
    return h @ rot.T if rot is not None else h


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shape_of(h) -> np.ndarray:
    h = np.asarray(h)
    return (h - h[0]) @ palm_frame(h) / np.linalg.norm(h[9] - h[0])


def test_never_seen_is_all_none():
    assert clean_track([None] * 5, 30) == [None] * 5


def test_gap_is_bridged_not_frozen():
    track = [hand(rot=rot_z(i * 3.0)).tolist() for i in range(20)]
    for i in range(8, 12):
        track[i] = None
    out = clean_track(track, 30)
    angle = [np.degrees(np.arctan2(*palm_frame(np.asarray(h))[[1, 0], 0])) for h in out]
    assert angle[7] < angle[9] < angle[10] < angle[12]  # still turning through the gap


def test_noisy_steady_handshape_is_held():
    rng = np.random.default_rng(0)
    track = [(hand(20) + rng.normal(0, 0.006, (21, 3))).tolist() for _ in range(40)]
    out = clean_track(track, 30)
    jitter_in = np.mean([np.abs(shape_of(a) - shape_of(b)).mean() for a, b in zip(track, track[1:])])
    jitter_out = np.mean([np.abs(shape_of(a) - shape_of(b)).mean() for a, b in zip(out, out[1:])])
    assert jitter_out < 0.4 * jitter_in


def test_real_handshape_change_is_kept():
    rng = np.random.default_rng(1)
    track = [(hand(0 if i < 20 else 80) + rng.normal(0, 0.004, (21, 3))).tolist() for i in range(40)]
    segs = segments(np.stack([shape_of(h) for h in track]), 30)
    assert len(segs) == 2 and abs(segs[1][0] - 20) <= 2
    out = clean_track(track, 30)
    pip_z = [shape_of(h)[6, 2] for h in out]  # index middle joint: in the palm plane when open, below it when curled
    assert pip_z[5] > -0.1 and pip_z[35] < -0.35


def test_orientation_follows_the_hand():
    track = [hand(30, rot_z(i * 4.0)).tolist() for i in range(30)]
    out = clean_track(track, 30)
    for i in (0, 15, 29):
        assert np.allclose(palm_frame(np.asarray(out[i])), palm_frame(np.asarray(track[i])), atol=0.05)


def test_knuckle_bend_amplified_toward_a_fist():
    h = fix_knuckles(hand(50))
    palm = h[9] - h[0]
    prox = h[10] - h[9]
    bend = np.degrees(np.arccos(prox @ palm / np.linalg.norm(prox) / np.linalg.norm(palm)))
    assert bend > 70  # 35 + (50 - 35) x 2.6
