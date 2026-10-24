import numpy as np

from server.citizen_classifier import _to_fps


def test_sparse_frames_are_resampled_to_30_fps_without_blending_a_missing_hand():
    t = np.arange(10) * 66.7  # the browser's ~15 frames/s
    data = np.zeros((10, 75, 2))
    data[:, :33] = np.arange(10)[:, None, None]  # pose moves steadily
    data[:5, 54:75] = 1.0  # right hand seen in the first half only
    out = _to_fps(data, t, 30.0)
    assert len(out) == 19
    assert np.allclose(out[1, :33], 0.5, atol=1e-3)  # pose interpolated between frames 0 and 1
    hand = out[:, 54:75].reshape(len(out), -1)
    assert set(np.unique(hand)) <= {0.0, 1.0}  # never a half-faded hand
