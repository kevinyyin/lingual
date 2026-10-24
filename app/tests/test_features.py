import numpy as np

from server.features import FEAT_DIM, frame_features, sequence_features
from frames import make_frame


def test_missing_hand_zero_and_flag():
    v = frame_features(make_frame(right=False))
    assert v.shape == (FEAT_DIM,)
    right = v[64:128]  # layout: left hand (63 + flag), right hand (63 + flag), ...
    assert np.all(right[:63] == 0) and right[63] == 0
    assert v[63] == 1


def test_scale_invariant():
    a = frame_features(make_frame())
    b = frame_features(make_frame(scale=2.0))
    assert np.allclose(a, b, atol=1e-5)


def test_resample_shape():
    frames = [make_frame(t=i) for i in range(17)]
    assert sequence_features(frames).shape == (32, FEAT_DIM)
    assert sequence_features(frames[:1]).shape == (32, FEAT_DIM)


def test_zero_shoulder_width_safe():
    v = frame_features(make_frame(shoulder_w=0.0))
    assert np.all(np.isfinite(v))
