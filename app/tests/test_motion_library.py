import json

from frames import make_frame
from training.build_motion_library import clip_json, smooth, trim_to_sign


def seq():
    """5 resting frames, 10 signing frames, 5 resting frames."""
    out = []
    for i in range(20):
        f = make_frame(t=i * 33.0)
        y = 0.5 if 5 <= i < 15 else 0.95
        f.pose[15][1] = f.pose[16][1] = y
        out.append(f)
    return out


def test_trim_keeps_only_raised_frames():
    assert trim_to_sign(seq()) == (5, 15)


def test_trim_never_raised_keeps_all():
    fs = seq()
    for f in fs:
        f.pose[15][1] = f.pose[16][1] = 0.95
    assert trim_to_sign(fs) == (0, 20)


def test_smooth_keeps_length_and_reduces_jitter():
    xs = [[[float(i % 2), 0.0, 0.0]] for i in range(10)]
    s = smooth(xs)
    assert len(s) == 10
    assert max(p[0][0] for p in s[2:-2]) < 0.9


def test_clip_json_shape():
    fs = seq()
    world = [[[0.0, 0.0, 0.0, 1.0]] * 33 for _ in fs]
    clip = json.loads(clip_json(fs, world, fps=30))
    assert clip["fps"] == 30 and len(clip["frames"]) == 10
    fr = clip["frames"][0]
    assert len(fr["pose"]) == 33 and len(fr["pose"][0]) == 3 and len(fr["pose2d"][0]) == 2
    assert fr["left"] is not None and len(fr["left"]) == 21
