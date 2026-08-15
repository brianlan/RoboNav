import torch
from prefusion.dataset.tensor_smith import TensorSmith

from robonav.aqua.transformable.goal import Goal
from robonav.registry import TENSOR_SMITHS


__all__ = ["GoalTensorSmith"]


@TENSOR_SMITHS.register_module()
class GoalTensorSmith(TensorSmith):
    def __call__(self, transformable: Goal):
        tensor_dict = dict(
            rotation=torch.tensor(transformable.rotation, dtype=torch.float32),
            translation=torch.tensor(transformable.translation, dtype=torch.float32),
        )
        if transformable.linear_velocity is not None:
            tensor_dict["linear_velocity"] = torch.tensor(
                transformable.linear_velocity, dtype=torch.float32
            )
        if transformable.angular_velocity is not None:
            tensor_dict["angular_velocity"] = torch.tensor(
                transformable.angular_velocity, dtype=torch.float32
            )
        return tensor_dict
