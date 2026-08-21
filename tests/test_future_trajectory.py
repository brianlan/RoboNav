import os

import numpy as np
import pytest
import torch
from mmengine.config import Config

from robonav.aqua.model_feeder.aqua_model_feeder import AquaModelFeeder
from robonav.aqua.tensor_smith.future_trajectory_tensor_smith import (
    FutureTrajectoryTensorSmith,
)
from robonav.aqua.transformable.future_trajectory import FutureTrajectory
from robonav.aqua.transformable_loader.future_trajectory_loader import (
    FutureTrajectoryLoader,
)
from robonav.registry import TENSOR_SMITHS, TRANSFORMABLE_LOADERS


def _frame_data(num_steps=2, with_velocities=True):
    data = {
        "rotation": [
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ]
        * num_steps,
        "translation": [[1, 2, 3], [11, 12, 13]][:num_steps],
    }
    if with_velocities:
        data["linear_velocity"] = [[4, 5, 6], [14, 15, 16]][:num_steps]
        data["angular_velocity"] = [[7, 8, 9], [17, 18, 19]][:num_steps]
    return {"future_trajectory": data}


def _load(frame_data=None, tensor_smith=None):
    return FutureTrajectoryLoader(data_root=None).load(
        "ft", None, frame_data or _frame_data(), None, tensor_smith=tensor_smith
    )


def test_future_trajectory_flip_3d():
    traj = _load().flip_3d(np.diag([1, -1, 1]))

    np.testing.assert_allclose(
        traj.rotation, [[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]] * 2
    )
    np.testing.assert_allclose(traj.translation, [[1, -2, 3], [11, -12, 13]])
    np.testing.assert_allclose(traj.linear_velocity, [[4, -5, 6], [14, -15, 16]])
    np.testing.assert_allclose(traj.angular_velocity, [[-7, 8, -9], [-17, 18, -19]])


def test_future_trajectory_rotate_3d():
    rmat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    traj = _load().rotate_3d(rmat)

    np.testing.assert_allclose(
        traj.rotation, [[[0, -1, 0], [0, 0, -1], [1, 0, 0]]] * 2
    )
    np.testing.assert_allclose(traj.translation, [[1, -3, 2], [11, -13, 12]])
    np.testing.assert_allclose(traj.linear_velocity, [[4, -6, 5], [14, -16, 15]])
    np.testing.assert_allclose(traj.angular_velocity, [[7, -9, 8], [17, -19, 18]])


def test_future_trajectory_translate_3d():
    traj = _load().translate_3d([0.5, 1, 1.5])

    np.testing.assert_allclose(
        traj.rotation, [[[0, -1, 0], [1, 0, 0], [0, 0, 1]]] * 2
    )
    np.testing.assert_allclose(traj.translation, [[0.5, 1, 1.5], [10.5, 11, 11.5]])
    np.testing.assert_allclose(traj.linear_velocity, [[4, 5, 6], [14, 15, 16]])


def test_constructor_validates_shapes_and_alignment():
    with pytest.raises(ValueError, match="rotation"):
        FutureTrajectory("ft", rotation=np.eye(3), translation=np.ones((2, 3)))
    with pytest.raises(ValueError, match="translation"):
        FutureTrajectory("ft", rotation=np.eye(3)[None], translation=np.ones((2, 4)))
    with pytest.raises(ValueError, match="inconsistent numbers of steps"):
        FutureTrajectory(
            "ft",
            rotation=np.zeros((2, 3, 3)),
            translation=np.ones((3, 3)),  # extra step would silently truncate
        )
    empty = FutureTrajectory(
        "ft", rotation=np.zeros((0, 3, 3)), translation=np.zeros((0, 3))
    )
    assert empty.rotation.shape == (0, 3, 3)


def test_loader_mapping_order_and_tensor_smith_propagation():
    smith = FutureTrajectoryTensorSmith()
    traj = _load(tensor_smith=smith)

    assert traj.name == "ft"
    assert traj.tensor_smith is smith
    np.testing.assert_allclose(traj.translation, [[1, 2, 3], [11, 12, 13]])  # order preserved
    np.testing.assert_allclose(traj.linear_velocity, [[4, 5, 6], [14, 15, 16]])
    np.testing.assert_allclose(traj.angular_velocity, [[7, 8, 9], [17, 18, 19]])

    traj.to_tensor()
    assert traj.tensor.shape == (2, 7)
    assert traj.tensor.dtype == torch.float32


def test_loader_raises_on_mismatched_field_lengths():
    frame_data = _frame_data()
    frame_data["future_trajectory"]["translation"].append([21, 22, 23])
    with pytest.raises(ValueError, match="inconsistent numbers of steps"):
        _load(frame_data)


def test_tensor_smith_values_dtypes_shapes():
    out = FutureTrajectoryTensorSmith()(_load())

    assert out.dtype == torch.float32
    assert out.shape == (2, 7)
    np.testing.assert_allclose(
        out.numpy(),
        [
            [1, 2, np.sin(np.pi / 2), np.cos(np.pi / 2), 4, 5, 9],
            [11, 12, np.sin(np.pi / 2), np.cos(np.pi / 2), 14, 15, 19],
        ],
        atol=1e-6,
    )


def test_reverse_recovers_yaw_and_layout():
    smith = FutureTrajectoryTensorSmith()
    yaw = torch.tensor([-3 * np.pi / 4, -np.pi / 2, np.pi / 3, np.pi])
    wrapped = (yaw.numpy() + np.pi) % (2 * np.pi) - np.pi
    forward = torch.stack(
        [torch.ones(4), torch.full((4,), 2.0), torch.sin(yaw), torch.cos(yaw),
         torch.full((4,), 4.0), torch.full((4,), 5.0), torch.full((4,), 6.0)],
        dim=-1,
    )
    assert forward.shape == (4, 7)

    out = smith.reverse(forward)
    assert out.shape == (4, 6)
    assert out.dtype == forward.dtype
    np.testing.assert_allclose(
        out.numpy(),
        [[1, 2, w, 4, 5, 6] for w in wrapped],
        atol=1e-6,
    )

    batched = smith.reverse(forward.unsqueeze(0))
    assert batched.shape == (1, 4, 6)
    np.testing.assert_allclose(batched.numpy()[0], out.numpy(), atol=1e-6)


def test_tensor_smith_requires_velocities():
    with pytest.raises(ValueError, match="requires linear_velocity and angular_velocity"):
        FutureTrajectoryTensorSmith()(_load(_frame_data(with_velocities=False)))


def test_config_wiring_and_registry_build():
    cfg = Config.fromfile(
        os.path.join(os.path.dirname(__file__), "..", "robonav", "aqua", "configs", "kinogoal_dla_resnet18_overfit.py")
    )
    ft_cfg = cfg.transformables["future_trajectory"]
    assert ft_cfg["type"] == "robonav.FutureTrajectory"
    assert ft_cfg["loader"]["type"] == "robonav.FutureTrajectoryLoader"
    assert ft_cfg["tensor_smith"]["type"] == "robonav.FutureTrajectoryTensorSmith"

    loader = TRANSFORMABLE_LOADERS.build({"type": ft_cfg["loader"]["type"], "data_root": None})
    smith = TENSOR_SMITHS.build({"type": ft_cfg["tensor_smith"]["type"]})
    traj = loader.load("ft", None, _frame_data(), None, tensor_smith=smith)
    traj.to_tensor()
    assert traj.tensor.shape == (2, 7)
    assert traj.tensor.dtype == torch.float32


def test_model_feeder_passes_trajectory_tensor():
    traj = _load(tensor_smith=FutureTrajectoryTensorSmith())
    traj.to_tensor()
    frame = {"index_info": {}, "transformables": {"future_trajectory": traj}}
    out = AquaModelFeeder()._process_transformables(frame)

    assert out["future_trajectory"] is traj.tensor
    assert not isinstance(out["future_trajectory"], FutureTrajectory)
    assert out["future_trajectory"].shape == (2, 7)
