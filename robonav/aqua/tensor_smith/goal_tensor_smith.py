import torch
from prefusion.dataset.tensor_smith import TensorSmith
from scipy.spatial.transform import Rotation

from robonav.aqua.transformable.goal import Goal
from robonav.registry import TENSOR_SMITHS


__all__ = ["GoalTensorSmith"]


@TENSOR_SMITHS.register_module()
class GoalTensorSmith(TensorSmith):
    def __call__(self, transformable: Goal):
        rot = Rotation.from_matrix(transformable.rotation).as_euler("XYZ", degrees=False)
        x, y = transformable.translation.flatten()[:2].tolist()
        vx, vy = transformable.linear_velocity.flatten()[:2].tolist()
        omega = transformable.angular_velocity.flatten()[-1]
        return torch.tensor([x, y, rot[2], vx, vy, omega], dtype=torch.float32)
