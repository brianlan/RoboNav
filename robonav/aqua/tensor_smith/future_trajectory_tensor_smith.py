import torch
import numpy as np
from scipy.spatial.transform import Rotation
from prefusion.dataset.tensor_smith import TensorSmith

from robonav.aqua.transformable.future_trajectory import FutureTrajectory
from robonav.registry import TENSOR_SMITHS


__all__ = ["FutureTrajectoryTensorSmith"]


@TENSOR_SMITHS.register_module()
class FutureTrajectoryTensorSmith(TensorSmith):
    def __call__(self, transformable: FutureTrajectory):
        if transformable.linear_velocity is None or transformable.angular_velocity is None:
            raise ValueError(
                "FutureTrajectoryTensorSmith requires linear_velocity and angular_velocity"
            )
        yaw = Rotation.from_matrix(transformable.rotation).as_euler("XYZ", degrees=False)[:, 2:3]
        xy = transformable.translation[:, :2]
        vx_vy = transformable.linear_velocity[:, :2]
        omega = transformable.angular_velocity[:, 2:3]
        return torch.tensor(
            np.concatenate([xy, np.sin(yaw), np.cos(yaw), vx_vy, omega], axis=1),
            dtype=torch.float32,
        )