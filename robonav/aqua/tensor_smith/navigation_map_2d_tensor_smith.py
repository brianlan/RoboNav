import numpy as np
import torch
import torch.nn.functional as F
from prefusion.dataset.tensor_smith import TensorSmith
from robonav.aqua.transformable import NavigationMap2D
from robonav.registry import TENSOR_SMITHS

__all__ = ["NavigationMap2DTensorSmith"]


@TENSOR_SMITHS.register_module()
class NavigationMap2DTensorSmith(TensorSmith):
    def __init__(self, output_range=(-1, -3, 5, 3), resolution=0.05):
        self.output_range = tuple(output_range)
        self.resolution = float(resolution)
        if len(self.output_range) != 4 or self.resolution <= 0:
            raise ValueError(
                "output_range must contain four values and resolution must be positive"
            )
        if any(self.output_range[i + 2] <= self.output_range[i] for i in (0, 1)):
            raise ValueError("output_range maxima must exceed minima")
        spans = [
            (self.output_range[i + 2] - self.output_range[i]) / self.resolution
            for i in (0, 1)
        ]
        if any(not np.isclose(span, round(span)) for span in spans):
            raise ValueError(
                "output_range spans must be integral multiples of resolution"
            )

    def __call__(self, transformable: NavigationMap2D):
        xmin, ymin, xmax, ymax = self.output_range
        nx, ny = (
            round((xmax - xmin) / self.resolution),
            round((ymax - ymin) / self.resolution),
        )
        i, j = torch.meshgrid(
            torch.arange(nx, dtype=torch.float64),
            torch.arange(ny, dtype=torch.float64),
            indexing="ij",
        )
        body = torch.stack(
            [
                xmin + (i + 0.5) * self.resolution,
                ymin + (j + 0.5) * self.resolution,
                torch.ones_like(i),
            ],
            -1,
        ).numpy()
        pixel = body @ np.linalg.inv(transformable.source_pixel_to_body).T
        h, w = transformable.occupancy.shape
        grid = torch.from_numpy(
            np.stack(
                [
                    pixel[..., 0] / (w - 1) * 2 - 1,
                    pixel[..., 1] / (h - 1) * 2 - 1,
                ],
                -1,
            )
        ).float()[None]

        def sample(array, mode):
            x = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())[None, None]
            return F.grid_sample(
                x, grid, mode=mode, padding_mode="zeros", align_corners=True
            )[0, 0]

        occ = sample(transformable.occupancy, "nearest")
        inside = (grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)
        occ = torch.where(inside, occ, torch.full_like(occ, 127))
        occupancy = torch.stack([(occ == 127), (occ == 255), (occ == 0)]).float()
        clearance = sample(transformable.clearance, "bilinear").where(
            inside, torch.zeros_like(occ)
        )[None]
        traversability = (sample(transformable.traversability, "nearest") > 0.5).where(
            inside, torch.zeros_like(occ, dtype=torch.bool)
        )[None]
        return {
            "occupancy": occupancy,
            "clearance": clearance,
            "traversability": traversability,
        }
