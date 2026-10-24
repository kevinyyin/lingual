import numpy as np
import torch

from frames import make_frame
from server.protocol import Frame, parse_client
from server.session import Segment
from server.spike_classifier import N_FEAT, SignNet, SpikeSignPredictor, frames_to_arrays, normalize


def checkpoint(tmp_path, classes):
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    torch.save({"state": SignNet(len(classes)).state_dict(), "classes": classes}, path)
    return path


def test_spike_names_mapped_and_topk_sorted(tmp_path):
    m = SpikeSignPredictor(checkpoint(tmp_path, ["WANT", "BYE", "IX-1P", "BANANA"]))
    assert m.labels == ["WANT", "GOODBYE", "I", "BANANA"]
    top = m.recognize(Segment(frames=[make_frame(t=i * 33.0) for i in range(20)], kind="sign"))
    assert len(top) == 3 and [p for _, p in top] == sorted((p for _, p in top), reverse=True)
    assert {g for g, _ in top} <= set(m.labels) and 0 < sum(p for _, p in top) <= 1.0 + 1e-6


def test_hand_assigned_to_the_wrist_it_sits_on():
    f = make_frame()
    sign_right = [[*f.pose[16][:2], 0.0]] * 21  # a hand on the signer's right wrist, whatever the browser called it
    frame = Frame(t=0, pose=f.pose, left=sign_right, right=None)
    lh, rh, _, _ = frames_to_arrays([frame])
    assert np.isnan(lh).all() and not np.isnan(rh).any()


def test_face_points_reach_the_features():
    f = make_frame()
    with_face = Frame(t=0, pose=f.pose, left=f.left, right=f.right, face=[[0.5, 0.3]] * 6)
    face = frames_to_arrays([with_face])[3]
    assert not np.isnan(face).any()
    assert normalize(*frames_to_arrays([with_face] * 5)).shape == (32, N_FEAT)


def test_frame_message_accepts_face_points():
    msg = parse_client('{"type": "frame", "t": 0, "pose": [[0, 0, 0, 1]], "face": [[0.1, 0.2]]}')
    assert msg.face == [[0.1, 0.2]]
