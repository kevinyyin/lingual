import numpy as np

from server.citizen_classifier import _to_fps, to_app_gloss


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


def test_renamed_signs_beat_extended_entries_with_the_citizen_name():
    # vocab_extended.yaml has its own BYE, ME and EXCUSE; ASL Citizen's classes of those names are our GOODBYE, I, EXCUSE-ME
    by_citizen = {"GOODBYE": "GOODBYE", "BYE": "BYE", "I": "I", "ME": "ME", "EXCUSEME": "EXCUSE-ME", "EXCUSE-ME": "EXCUSE-ME",
                  "EXCUSE": "EXCUSE", "WANT": "WANT", "COFFEE": "COFFEE",
                  "": "WATERFALL"}  # WLASL-only signs have no ASL Citizen name (609 of them)
    assert to_app_gloss("COFFEE", by_citizen) == "COFFEE"
    assert to_app_gloss("BYE", by_citizen) == "GOODBYE"
    assert to_app_gloss("ME", by_citizen) == "I"
    assert to_app_gloss("EXCUSE", by_citizen) == "EXCUSE-ME"
    assert to_app_gloss("WANT2", by_citizen) == "WANT"  # variants still merge
    assert to_app_gloss("BYE", {"BYE": "BYE"}) == "BYE"  # no GOODBYE in the vocabulary: keep the citizen name


def test_the_translator_reads_only_the_hand_checked_everyday_signs():
    import pytest

    from frames import make_frame
    import server.main as M
    from server.vocab import load_vocab

    if not hasattr(M.RECOGNIZER, "labels"):
        pytest.skip("no sign model installed")
    curated = load_vocab(extended=None).glosses
    frames = [make_frame(t=i * 66.0) for i in range(20)]
    possible = {g for g, p in M.RECOGNIZER.topk(frames, 10**6) if p > 0}
    assert possible and possible <= curated, sorted(possible - curated)[:5]
