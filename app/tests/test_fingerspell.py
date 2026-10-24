import numpy as np

from frames import make_frame
from server.fingerspell import LETTERS, WINDOW, LetterNet, SpellReader, decode, hand_features, save_letter_model
from server.session import Segment


def probs(seq: str, p: float = 0.9) -> np.ndarray:
    """One row per window; '.' is a low-confidence window."""
    out = np.full((len(seq), 26), (1 - p) / 25)
    for i, ch in enumerate(seq):
        if ch == ".":
            out[i] = 1 / 26
        else:
            out[i, LETTERS.index(ch)] = p
    return out


def test_collapse_repeats():
    assert decode(probs("AAANNAA"))[0][0] == "ANA"


def test_low_confidence_windows_dropped():
    assert decode(probs("AA..BB"))[0][0] == "AB"


def test_double_letter_with_dip():
    assert decode(probs("LL.LL"))[0][0] == "LL"


def test_alternatives_change_least_confident_letter():
    p = probs("CCAATT")
    p[2:4, LETTERS.index("A")] = 0.55
    p[2:4, LETTERS.index("U")] = 0.4
    words = [w for w, _ in decode(p)]
    assert words[0] == "CAT" and "CUT" in words
    assert len(words) == len(set(words))


def test_empty_when_nothing_confident():
    assert decode(probs("....")) == []


def test_hand_features_scale_invariant():
    f = make_frame()
    a = hand_features(f.right)
    b = hand_features([[x * 2, y * 2, z * 2] for x, y, z in f.right])
    assert a.shape == (63,) and np.allclose(a, b, atol=1e-5)


def test_spell_reader_returns_words(tmp_path):
    p = tmp_path / "letters.pt"
    save_letter_model(LetterNet(), p)
    seg = Segment(frames=[make_frame(t=i, left=False) for i in range(WINDOW + 4)], kind="spell")
    out = SpellReader(p).recognize(seg)
    assert isinstance(out, list) and all(isinstance(w, str) and 0 <= pr <= 1 for w, pr in out)


def test_letter_dataset_and_clip(tmp_path):
    import json

    from training.landmarks import save_npz
    from training.train_fingerspell import letter_clip, letter_dataset

    for signer in ("kev", "ana"):
        for ch in "AB":
            save_npz([make_frame(t=i, left=False, jitter=0.001 * i) for i in range(20)], tmp_path / signer / "_letters" / ch / "0.npz")
    X, y = letter_dataset(tmp_path, ["kev"])
    assert X.shape[1] == WINDOW * 63 and set(y.tolist()) == {0, 1}
    X2, _ = letter_dataset(tmp_path, ["kev", "ana"])
    assert len(X2) == 2 * len(X)
    clip = json.loads(letter_clip([make_frame(t=i, left=False) for i in range(20)]))
    fr = clip["frames"][0]
    assert len(fr["pose"]) == 33 and len(fr["pose2d"][0]) == 2 and fr["right"] is not None
