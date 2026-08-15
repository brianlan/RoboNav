import numpy as np
from prefusion.dataset.transformable_loader import TransformableLoader
from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import PolarDict

from robonav.aqua.transformable import Goal
from robonav.registry import TRANSFORMABLE_LOADERS


__all__ = ["GoalLoader"]


@TRANSFORMABLE_LOADERS.register_module()
class GoalLoader(TransformableLoader):
    def load(
        self,
        name: str,
        frame_info: PolarDict,
        frame_data: dict[str, dict],
        index_info: IndexInfo,
        tensor_smith: TensorSmith | None = None,
        **kwargs,
    ) -> Goal:
        goal_data = frame_data["goal"]
        return Goal(
            name,
            rotation=np.array(goal_data["rotation"]),
            translation=goal_data["translation"],
            tensor_smith=tensor_smith,
            linear_velocity=goal_data.get("linear_velocity"),
            angular_velocity=goal_data.get("angular_velocity"),
        )
