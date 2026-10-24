import json

import numpy as np

from training.build_motion_library import clip_json
from training.clips_from_spike import _match, npz_frames


def spike_npz(path, T=12):
    """Holistic-style npz as asl-spike's s2_extract writes it: the signer's right wrist (pose 16) raised mid-clip."""
    pose = np.zeros((T, 33, 4), np.float32)
    pose[:, :, 3] = 1.0
    pose[:, 15, :2] = (0.65, 0.9)  # signer's left wrist (image right on unmirrored video), at rest
    pose[:, 16, :2] = (0.35, 0.9)
    pose[3:9, 16, 1] = 0.5  # right hand up for the sign
    world = np.zeros((T, 33, 3), np.float32)
    rh = np.zeros((T, 21, 3), np.float32)  # Holistic's label for the hand on the signer's right wrist
    rh[:, :, :2] = pose[:, 16:17, :2]
    lh = np.full((T, 21, 3), np.nan, np.float32)
    pose[0, 0, 0] = np.nan  # a frame with no person is dropped
    np.savez_compressed(path, pose=pose, pose_world=world, lh=lh, rh=rh, face=np.zeros((T, 1, 3)), blend=np.zeros((T, 52)),
                        fps=25.0, n_frames=T, width=640, height=480, extract_ms=0.0)


def test_hands_keyed_by_the_signers_own_side(tmp_path):
    spike_npz(tmp_path / "x.npz")
    frames, world, fps = npz_frames(tmp_path / "x.npz")
    assert fps == 25.0 and len(frames) == len(world) == 11
    assert all(f.right is not None and f.left is None for f in frames)  # the hand on the signer's right wrist


def test_clip_trimmed_to_the_sign(tmp_path):
    spike_npz(tmp_path / "x.npz")
    clip = json.loads(clip_json(*npz_frames(tmp_path / "x.npz")))
    assert clip["fps"] == 25.0 and len(clip["frames"]) == 6  # only the raised frames
    f = clip["frames"][0]
    assert len(f["pose"]) == 33 and len(f["pose2d"]) == 33 and len(f["right"]) == 21 and f["left"] is None


def test_citizen_variants_and_aliases_match():
    want = {"WANT": "WANT", "I": "I", "GOODBYE": "GOODBYE", "WHAT": "WHAT", "THANKYOU": "THANK-YOU"}
    assert _match("WANT1", want) == ("WANT", 0)
    assert _match("WANT2", want) == ("WANT", 1)  # variant 1 preferred
    assert _match("W.H.A.T", want) == ("WHAT", 2)  # spelled form last
    assert _match("THANKYOU", want) == ("THANK-YOU", 0)
    assert _match("ME", want)[0] == "I" and _match("BYE", want)[0] == "GOODBYE"
    assert _match("BANANA1", want)[0] is None


def test_signs2_entry_becomes_a_clip():
    """asl-spike avatar/signs2.json layout: pw/p2/lh/rh per frame, null where missing."""
    from training.clips_from_spike import signs2_frames

    def fr(raised, hand=True):
        p2 = [[0.5, 0.3, 0.0, 1.0] for _ in range(33)]
        p2[11], p2[12] = [0.6, 0.5, 0.0, 1.0], [0.4, 0.5, 0.0, 1.0]
        p2[15], p2[16] = [0.65, 0.9, 0.0, 1.0], [0.35, 0.5 if raised else 0.9, 0.0, 1.0]
        rh = [[p2[16][0], p2[16][1], 0.0]] * 21 if hand else None  # on the signer's right wrist
        return {"pw": [[0.0, 0.0, 0.0]] * 33, "p2": p2, "lh": None, "rh": rh}

    entry = {"fps": 25.0, "frames": [fr(False)] * 3 + [fr(True)] * 5 + [fr(True, hand=False)] + [fr(False)] * 3
             + [{"pw": None, "p2": None, "lh": None, "rh": None}]}
    frames, world, fps = signs2_frames(entry)
    assert fps == 25.0 and len(frames) == len(world) == 12  # the all-null frame is dropped
    assert frames[3].right is not None and frames[3].left is None  # the hand on the signer's right wrist
    clip = json.loads(clip_json(frames, world, fps))
    assert len(clip["frames"]) == 6 and clip["frames"][-1]["right"] is not None  # raised span; hand gap filled


def test_clip_rekeys_a_flipped_detector_label_by_wrist():
    """MediaPipe flips handedness labels on ~22% of frames; clip_json must follow the wrist, not the label."""
    from server.protocol import Frame
    from training.build_motion_library import by_wrist

    pose = [[0.5, 0.3, 0.0, 1.0] for _ in range(33)]
    pose[15], pose[16] = [0.65, 0.5, 0.0, 1.0], [0.35, 0.5, 0.0, 1.0]  # signer's left wrist is on image right
    on_signers_left = [[0.65, 0.5, 0.0]] * 21
    img, wld = by_wrist(Frame(t=0, pose=pose, left=None, right=on_signers_left), {"left": None, "right": [[1, 1, 1]] * 21})
    assert img["left"] == on_signers_left and img["right"] is None
    assert wld["left"] == [[1, 1, 1]] * 21  # the world hand follows its image hand
