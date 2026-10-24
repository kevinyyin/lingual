import numpy as np
import torch

from server.fingerspell_ctc import FEAT, CtcSpellReader, SpellNet, features, greedy
from server.protocol import Frame
from server.session import Segment
from training.train_fingerspell_ctc import collate, cer


def hands(T=12, side="right"):
    rng = np.random.default_rng(0)
    h = rng.random((T, 21, 3)).astype(np.float32)
    nan = np.full((T, 21, 3), np.nan, np.float32)
    return (h, nan) if side == "right" else (nan, h)


def test_features_mirror_a_left_speller_into_the_right_slot():
    r = features(*hands(side="right"))
    l = features(*hands(side="left"))
    assert r.shape == l.shape == (12, FEAT)
    assert np.allclose(r[:, 63:126], 0) and np.allclose(l[:, 63:126], 0)  # the other hand slot is empty
    # columns are point*3 + axis; the wrist (point 0) is the origin, so compare points 1-20
    assert np.allclose(r[:, 3:63:3], -l[:, 3:63:3]) and np.allclose(r[:, 4:63:3], l[:, 4:63:3])  # x mirrored, y kept
    assert (r[:, -2] == 1).all() and (l[:, -2] == 1).all()


def test_frames_without_hands_are_dropped():
    r, l = hands()
    r[3:6] = np.nan
    assert len(features(r, l)) == 9


def test_greedy_ctc_collapses_repeats_and_blanks():
    chars = ["a", "b", "c"]
    path = [1, 1, 0, 1, 2, 2, 0, 0, 3]  # a a _ a b b _ _ c -> "aabc"
    logp = np.log(np.full((len(path), 4), 1e-3))
    for t, k in enumerate(path):
        logp[t, k] = np.log(0.997)
    text, conf = greedy(logp, chars)
    assert text == "aabc" and conf > 0.99


def test_training_steps_reduce_loss():
    torch.manual_seed(0)
    items = [(features(*hands(T)), np.array([1, 2, 3]), "abc") for T in (10, 14, 20)]
    model = SpellNet(5, d=32, layers=1, heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    losses = []
    for _ in range(25):
        X, pad, xl, Y, yl = collate(items, [0, 1, 2])
        loss = ctc(torch.log_softmax(model(X, pad), -1).transpose(0, 1), Y, xl, yl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_reader_returns_letters_only(tmp_path):
    chars = ["A", "B", " ", "1"]
    torch.save({"state": SpellNet(len(chars)).state_dict(), "chars": chars}, tmp_path / "m.pt")
    reader = CtcSpellReader(tmp_path / "m.pt")
    pose = [[0.5, 0.5, 0, 1]] * 33
    seg = Segment(frames=[Frame(t=i, pose=pose, right=np.random.rand(21, 3).tolist()) for i in range(10)], kind="spell")
    out = reader.recognize(seg)
    assert all(w.isalpha() and w.isupper() for w, _ in out)


def test_cer():
    assert cer("ANA", "ANA") == 0 and cer("ANA", "") == 1 and abs(cer("ANNA", "ANA") - 0.25) < 1e-9
