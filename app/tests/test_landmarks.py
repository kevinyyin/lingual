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


def test_one_zip_member_is_read_with_range_requests_only(tmp_path):
    import zipfile
    from training.remote_zip import member_bytes

    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ASL_Citizen/videos/a.mp4", b"stored bytes" * 50, compress_type=zipfile.ZIP_STORED)
        z.writestr("ASL_Citizen/videos/b.mp4", b"deflated bytes" * 50, compress_type=zipfile.ZIP_DEFLATED)
    raw = path.read_bytes()
    asked = []

    def fetch(lo, hi):  # inclusive byte range, as an HTTP Range header
        asked.append((lo, hi))
        return raw[lo:hi + 1]

    with zipfile.ZipFile(path) as z:
        for info, want in zip(z.infolist(), (b"stored bytes" * 50, b"deflated bytes" * 50)):
            assert member_bytes(fetch, info) == want
    assert all(hi - lo < 1000 for lo, hi in asked)  # never the whole file


def test_select_rows_merges_numbered_variants_and_takes_asl_citizens_other_names(tmp_path):
    from training.extract_asl_citizen import select_rows

    csv = tmp_path / "train.csv"
    csv.write_text("Participant ID,Video file,Gloss,ASL-LEX Code\n"
                   "P1,a.mp4,WANT1,x\nP2,b.mp4,WANT2,y\nP3,c.mp4,ME,z\nP4,d.mp4,WANTED,w\n")
    rows = select_rows(csv, {"WANT": "WANT", "I": "I"})
    assert rows == [("P1", "a.mp4", "WANT"), ("P2", "b.mp4", "WANT"), ("P3", "c.mp4", "I")]
