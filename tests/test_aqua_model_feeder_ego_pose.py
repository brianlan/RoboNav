import numpy as np
import torch
from prefusion.dataset.transform import EgoPose, EgoPoseSet

from robonav.aqua.model_feeder.aqua_model_feeder import AquaModelFeeder


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_pose(rel_pos, yaw, translation, lin_vel=None, ang_vel=None):
    return EgoPose(
        f"ego_pose:{rel_pos}",
        rel_pos,
        rot_z(yaw),
        np.asarray(translation, dtype=np.float64).reshape(3, 1),
        linear_velocity=lin_vel,
        angular_velocity=ang_vel,
    )


def process_egopose_set(ego_pose_set):
    frame = {"transformables": {"ego_pose": ego_pose_set}}
    return AquaModelFeeder()._process_transformables(frame)


def test_twist_delta_pose_and_no_previous_pose():
    yaw1, delta_yaw = 0.3, 0.5
    t_w_e1 = np.array([1.0, 2.0, 0.1])
    # place e2 so the e1-frame displacement is exactly (0.7, -0.4, 0)
    t_w_e2 = t_w_e1 + rot_z(yaw1) @ np.array([0.7, -0.4, 0.0])
    lin = [1.5, -0.2, 0.0]
    ang = [0.1, 0.2, 0.6]

    prev = make_pose("-1", yaw1, t_w_e1, lin, ang)
    cur = make_pose("0", yaw1 + delta_yaw, t_w_e2, lin, ang)
    out = process_egopose_set(EgoPoseSet("ego_pose", {"-1": prev, "0": cur}))

    torch.testing.assert_close(out["twist"], torch.tensor([1.5, -0.2, 0.6]))
    torch.testing.assert_close(
        out["delta_poses"], torch.tensor([0.7, -0.4, delta_yaw]), atol=1e-6, rtol=0
    )
    assert out["twist"].shape == (3,) and out["twist"].dtype == torch.float32
    assert out["delta_poses"].shape == (3,) and out["delta_poses"].dtype == torch.float32

    # a batch of frames stacks to (B, 3)
    twist_b = torch.stack([out["twist"], out["twist"]])
    assert twist_b.shape == (2, 3)

    # no previous pose ("-1" absent): zero delta pose
    out_no_prev = process_egopose_set(EgoPoseSet("ego_pose", {"0": cur}))
    assert out_no_prev["delta_poses"].dtype == torch.float32
    assert out_no_prev["delta_poses"].shape == (3,)
    assert torch.all(out_no_prev["delta_poses"] == 0)
