import pytest

from frames import make_frame
from server.protocol import RecordMsg
from server.record import save_recording
from server.vocab import load_vocab
from training.landmarks import load_npz

V = load_vocab(extended=None)  # curated only: tests must not depend on the generated extension


def msg(signer="kevin", gloss="COFFEE", n=5):
    return RecordMsg(signer=signer, gloss=gloss, frames=[make_frame(t=i) for i in range(n)])


def test_record_writes_npz_to_signer_gloss_dir(tmp_path):
    p1 = save_recording(tmp_path, msg(), V)
    p2 = save_recording(tmp_path, msg(), V)
    assert p1 == tmp_path / "kevin" / "COFFEE" / "0.npz"
    assert p2.name == "1.npz"
    assert len(load_npz(p1)) == 5


def test_record_grid_and_letters(tmp_path):
    assert save_recording(tmp_path, msg(gloss="_grid/yn_happy"), V) == tmp_path / "kevin" / "_grid" / "yn_happy_0.npz"
    assert save_recording(tmp_path, msg(gloss="_letter/A"), V) == tmp_path / "kevin" / "_letters" / "A" / "0.npz"


def test_record_rejects_unknown_gloss(tmp_path):
    with pytest.raises(ValueError):
        save_recording(tmp_path, msg(gloss="ZEBRA"), V)
    with pytest.raises(ValueError):
        save_recording(tmp_path, msg(gloss="_grid/yn_bored"), V)


def test_record_rejects_path_traversal_signer(tmp_path):
    with pytest.raises(ValueError):
        save_recording(tmp_path, msg(signer="../x"), V)
    assert not any(tmp_path.parent.glob("x"))
    assert list(tmp_path.rglob("*.npz")) == []


def test_record_rejects_empty_take(tmp_path):
    with pytest.raises(ValueError):
        save_recording(tmp_path, msg(n=0), V)


def test_record_names(tmp_path):
    assert save_recording(tmp_path, msg(gloss="_name/PRIYA"), V) == tmp_path / "kevin" / "_names" / "PRIYA_0.npz"
    with pytest.raises(ValueError):
        save_recording(tmp_path, msg(gloss="_name/TOOLONGNAME"), V)
