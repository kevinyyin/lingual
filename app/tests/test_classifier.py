import numpy as np
import torch

from frames import make_frame
from server.classifier import SignNet, SignPredictor, save_checkpoint
from server.features import FEAT_DIM


def frames(n=12):
    return [make_frame(t=i * 33.0, jitter=0.002 * i) for i in range(n)]


def test_predictor_output_shape(tmp_path):
    labels = ["A", "B", "C", "D", "E"]
    p = tmp_path / "signs.pt"
    save_checkpoint(SignNet(len(labels)), labels, p)
    top = SignPredictor(p).topk(frames(), 3)
    assert len(top) == 3
    probs = [pr for _, pr in top]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-6
    assert {g for g, _ in top} <= set(labels)


def test_labels_come_from_checkpoint(tmp_path):
    p = tmp_path / "signs.pt"
    save_checkpoint(SignNet(2), ["ZEBRA", "YAK"], p)
    assert {g for g, _ in SignPredictor(p).topk(frames(), 2)} == {"ZEBRA", "YAK"}


def test_recognize_uses_segment_frames(tmp_path):
    from server.session import Segment

    p = tmp_path / "signs.pt"
    save_checkpoint(SignNet(3), ["A", "B", "C"], p)
    assert len(SignPredictor(p).recognize(Segment(frames=frames(), kind="sign"))) == 3


def test_training_learns_separable_classes(tmp_path):
    from training.train_signs import fit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 32, FEAT_DIM)).astype(np.float32) * 0.1
    y = np.arange(120) % 3
    X[:, :, 0] += y[:, None] * 1.0  # class signal
    model = fit(X[:90], y[:90], X[90:], y[90:], n_classes=3, epochs=15, device="cpu", use_aug=False)
    with torch.no_grad():
        acc = (model(torch.from_numpy(X[90:])).argmax(1).numpy() == y[90:]).mean()
    assert acc > 0.9


def test_augment_keeps_flags_and_missing_hand_zero():
    from server.features import sequence_features
    from training.train_signs import augment

    torch.manual_seed(0)
    x = torch.from_numpy(np.stack([sequence_features([make_frame(t=i, right=False) for i in range(10)])] * 16))
    out = augment(x)
    flags = out[..., [63, 127]]
    assert set(flags.unique().tolist()) <= {0.0, 1.0}
    # each sample has exactly one hand present, and the absent hand's block is all zero
    for o in out:
        present = 0 if o[0, 63] == 1 else 1
        absent = slice(64, 128) if present == 0 else slice(0, 64)
        assert torch.all(o[:, absent] == 0)


def test_swap_mirrors_symmetric_pose_onto_itself():
    from server.features import frame_features
    from training.train_signs import SWAP, TRIPLES

    v = frame_features(make_frame()).copy()
    m = v.copy()
    m[TRIPLES[:, 0]] *= -1
    m = m[SWAP]
    assert np.allclose(m[134:155], v[134:155], atol=1e-6)  # make_frame's pose is left/right symmetric


def _clip(value, n_rest=5, n_sign=10):
    """Per-frame features: rest (no hands) then a sign whose right hand carries `value`."""
    from server.features import FEAT_DIM
    x = np.zeros((n_rest + n_sign + n_rest, FEAT_DIM), np.float32)
    x[n_rest:n_rest + n_sign, 64:127] = value
    x[n_rest:n_rest + n_sign, 127] = 1.0
    return x


def test_a_sign_in_a_sentence_sits_between_its_neighbours_signs_and_is_cut_with_some_of_them():
    from training.train_signs import hand_span, in_sentence
    target, prev, nxt = _clip(5.0), _clip(1.0), _clip(9.0)
    assert hand_span(target) == (5, 15)
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = in_sentence(target, prev, nxt, rng, trans=3, max_ctx=6)
        vals = w[:, 64]
        assert (vals == 5.0).sum() == 10  # the whole target sign
        assert set(np.unique(vals)) <= {1.0, 5.0, 9.0} | set(vals[(vals > 1) & (vals < 9)])  # neighbours and blends only
        assert (vals != 5.0).sum() <= 2 * (6 + 3)  # at most max_ctx frames into each side, transitions included
        assert (w[:, 127] > 0).all()  # no rest inside a sentence


def test_without_neighbours_the_sign_keeps_its_own_rest():
    from training.train_signs import in_sentence
    w = in_sentence(_clip(5.0), None, None, np.random.default_rng(1), max_ctx=5)
    assert (w[:, 64] == 5.0).sum() == 10 and (w[:, 127] == 0).sum() <= 10


def test_blending_between_signs_never_drags_a_missing_hand_from_zero():
    from training.train_signs import blend
    a, b = _clip(2.0)[7], _clip(4.0)[7]
    b[64:128] = 0.0  # no right hand in the next sign
    out = blend(a, b, 4)
    assert set(np.unique(out[:, 64])) == {0.0, 2.0}  # the nearer side's hand, not 1.5 or 0.5
