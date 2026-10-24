import numpy as np

from training.audit_placement import aspect_from_pose, placement_error


def _frame(aspect=16 / 9):
    # world (x right-positive as MediaPipe's left shoulder, y down), metres; nose above shoulders, wrists by the face
    pose = np.zeros((33, 3))
    pose[0] = [0.0, -0.25, 0.0]
    pose[11], pose[12] = [0.2, 0.0, 0], [-0.2, 0.0, 0]
    pose[23], pose[24] = [0.15, 0.5, 0], [-0.15, 0.5, 0]
    pose[15], pose[16] = [0.3, -0.3, 0.1], [-0.2, -0.1, 0.2]
    # perspective-free projection: 2D normalised, x by width, y by height; 0.9 H per metre of height
    s = 0.9
    pose2d = np.stack([0.5 + pose[:, 0] * s / aspect, 0.5 + pose[:, 1] * s], axis=1)
    return {"pose": pose.tolist(), "pose2d": pose2d.tolist()}


def test_aspect_from_pose_recovers_ratio():
    f = _frame()
    assert abs(aspect_from_pose(f["pose"], f["pose2d"]) - 16 / 9) < 0.01


def test_placement_error_zero_when_consistent():
    f = _frame()
    err = placement_error(f, aspect_from_pose(f["pose"], f["pose2d"]))
    assert max(err.values()) < 1e-6
