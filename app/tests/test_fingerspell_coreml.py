import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from frames import make_frame
from server.fingerspell_coreml import DEFAULT_MODEL, decode, holistic, resample, standardize
from server.session import Segment
from training.train_fingerspell_ctc import cer

KAGGLE = Path(__file__).resolve().parent.parent / "data" / "kaggle_fs"


def test_each_hand_goes_to_the_slot_of_its_nearest_pose_wrist():
    f = make_frame(left=False)
    f.left, f.right = f.right, None  # the tracker labelled the signer's right hand "left" (4% of frames)
    f.mesh = [[0.5, 0.4, 0.01]] * 468
    x = holistic([f])
    assert x.shape == (1, 543, 3)
    assert np.isnan(x[0, 468:489]).all()  # the left-hand slot is empty
    assert np.allclose(x[0, 522:543], f.left)  # the hand at the signer's right wrist (pose 16) is the right hand
    assert np.allclose(x[0, 489:522], [p[:3] for p in f.pose]) and np.allclose(x[0, :468], f.mesh)


def test_frames_without_a_face_mesh_leave_the_face_missing():
    assert np.isnan(holistic([make_frame()])[0, :468]).all()


def test_frames_are_resampled_to_30_per_second_by_time():
    x = np.zeros((3, 543, 3), np.float32)
    x[:, 0, 0] = [0.0, 1.0, 2.0]
    x[2, 1] = np.nan  # a point missing on one side: the nearer frame's value, not NaN
    x[1, 1] = 5.0
    y = resample(x, np.array([0.0, 66.7, 133.4]))
    assert len(y) == 5 and np.allclose(y[:, 0, 0], [0, 0.5, 1, 1.5, 2], atol=0.01)
    assert y[3, 1, 0] == 5.0 and np.isnan(y[4, 1, 0])


def test_each_axis_is_standardised_over_the_window_and_missing_points_become_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (40, 543, 3)).astype(np.float32)
    x[:, :10] = np.nan
    y = standardize(x)
    assert np.allclose(y[:, 10:].reshape(-1, 3).mean(0), 0, atol=1e-4)
    assert np.allclose(y[:, 10:].reshape(-1, 3).std(0), 1, atol=1e-3)
    assert (y[:, :10] == 0).all()


def test_decode_keeps_letters_collapses_repeats_and_scores_confidence():
    vocab = ["<blank>", "1", "a", "b"]
    path = [2, 2, 0, 2, 1, 3, 3, 0]  # a a _ a 1 b b _ -> "aab"; the digit is never chosen
    logits = np.full((len(path), 4), -5.0, np.float32)
    logits[np.arange(len(path)), path] = 5.0
    logits[4, 0] = 4.0  # where the digit wins, the blank is next best
    text, conf = decode(logits, vocab)
    assert text == "aab" and 0.9 < conf <= 1.0


@pytest.mark.skipif(importlib.util.find_spec("coremltools") is None or not DEFAULT_MODEL.exists()
                    or not list((KAGGLE / "train_landmarks").rglob("*.parquet")) or importlib.util.find_spec("pyarrow") is None,
                    reason="needs coremltools, the Core ML model and a Kaggle landmark file")
def test_reads_a_real_spelled_phrase_through_app_frames():
    import pandas as pd
    from server.fingerspell_coreml import CoreMLSpellReader
    from server.protocol import Frame

    parquet = next((KAGGLE / "train_landmarks").rglob("*.parquet"))
    phrases = {r["sequence_id"]: r["phrase"] for r in csv.DictReader(open(KAGGLE / "train.csv"))}
    df = pd.read_parquet(parquet)
    seq = next(s for s in dict.fromkeys(df.index) if phrases[str(s)].replace(" ", "").isalpha())
    d = df.loc[seq]
    pts = lambda part, n: np.stack([d[[f"{a}_{part}_{i}" for i in range(n)]].to_numpy() for a in "xyz"], -1)
    face, left, pose, right = pts("face", 468), pts("left_hand", 21), pts("pose", 33), pts("right_hand", 21)
    frames = [Frame(t=i * 33.3, pose=[[*p, 1.0] for p in pose[i]], mesh=None if np.isnan(face[i]).all() else face[i].round(3).tolist(),
                    left=None if np.isnan(left[i]).all() else left[i].tolist(), right=None if np.isnan(right[i]).all() else right[i].tolist())
              for i in range(0, len(d), 2)]  # the browser's ~15 frames/s
    (word, conf), = CoreMLSpellReader().recognize(Segment(frames=frames, kind="spell"))
    ref = phrases[str(seq)].replace(" ", "").upper()
    assert cer(ref, word) <= 0.3, (word, ref)  # TONY BAUTISTA read as TONBARTISTA: 0.17
