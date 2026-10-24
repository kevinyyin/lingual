from frames import make_frame
from training.landmarks import load_npz, save_npz


def test_npz_roundtrip_preserves_missing_hand(tmp_path):
    frames = [make_frame(t=i * 33.0, left=(i % 2 == 0)) for i in range(5)]
    frames[1].blend = {"browInnerUp": 0.3}
    frames[1].head = (1.0, 2.0, 3.0)
    p = tmp_path / "x.npz"
    save_npz(frames, p)
    back = load_npz(p)
    assert [f.left is None for f in back] == [f.left is None for f in frames]
    assert back[1].blend == {"browInnerUp": 0.3} and back[0].blend is None
    assert back[1].head == (1.0, 2.0, 3.0)
    assert back[2].right == frames[2].right and back[3].t == frames[3].t


def test_select_rows_matches_vocab_case_insensitively(tmp_path):
    from training.extract_asl_citizen import select_rows

    csv = tmp_path / "train.csv"
    csv.write_text("Participant ID,Video file,Gloss,ASL-LEX Code\n"
                   "P1,a.mp4,hello,x\nP2,b.mp4,THANKYOU ,y\nP3,c.mp4,ZEBRA,z\n")
    rows = select_rows(csv, {"HELLO": "HELLO", "THANKYOU": "THANK-YOU"})
    assert rows == [("P1", "a.mp4", "HELLO"), ("P2", "b.mp4", "THANK-YOU")]
