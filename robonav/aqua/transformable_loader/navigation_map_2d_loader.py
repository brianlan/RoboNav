from pathlib import Path

import numpy as np
from PIL import Image
from prefusion.dataset.transformable_loader import TransformableLoader

from robonav.aqua.transformable import NavigationMap2D
from robonav.registry import TRANSFORMABLE_LOADERS

__all__ = ["NavigationMap2DLoader"]


@TRANSFORMABLE_LOADERS.register_module()
class NavigationMap2DLoader(TransformableLoader):
    def load(self, name, frame_info, frame_data, index_info, tensor_smith=None, **kwargs):
        info = frame_data["scene_info"]["navigation_map_2d"]
        root = Path(self.data_root)

        def path(key):
            return root / info[key]

        occupancy = np.asarray(Image.open(path("occupancy_path")).convert("L"))
        clearance = np.load(path("clearance_path"), allow_pickle=False)
        traversability = np.asarray(
            Image.open(path("traversability_path")).convert("L")
        )
        if not np.isin(traversability, (0, 255)).all():
            raise ValueError("traversability raster must contain only 0 and 255")
        traversability = traversability == 255
        ego = frame_data["ego_pose"]
        R = np.asarray(ego["rotation"], dtype=np.float64)
        t = np.asarray(ego["translation"], dtype=np.float64)
        m = np.asarray(info["pixel_to_world"], dtype=np.float64)
        world_to_body = np.eye(3)
        world_to_body[:2, :2] = R[:2, :2].T
        world_to_body[:2, 2] = -(R[:2, :2].T @ t[:2])
        return NavigationMap2D(
            name,
            occupancy,
            clearance,
            traversability,
            world_to_body @ m,
            tensor_smith,
        )
