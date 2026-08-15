from prefusion.dataset.transformable_loader import TransformableLoader
from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import PolarDict

from robonav.aqua.transformable import FutureTrajectory
from robonav.registry import TRANSFORMABLE_LOADERS


__all__ = ["FutureTrajectoryLoader"]


@TRANSFORMABLE_LOADERS.register_module()
class FutureTrajectoryLoader(TransformableLoader):
    def load(
        self,
        name: str,
        frame_info: PolarDict,
        frame_data: dict[str, dict],
        index_info: IndexInfo,
        tensor_smith: TensorSmith | None = None,
        **kwargs,
    ) -> FutureTrajectory:
        traj_data = frame_data["future_trajectory"]
        return FutureTrajectory(
            name,
            rotation=traj_data["rotation"],
            translation=traj_data["translation"],
            tensor_smith=tensor_smith,
            linear_velocity=traj_data.get("linear_velocity"),
            angular_velocity=traj_data.get("angular_velocity"),
        )
