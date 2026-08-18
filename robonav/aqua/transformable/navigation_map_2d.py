import numpy as np
from prefusion.dataset.transform import SpatialTransformable

from robonav.registry import TRANSFORMABLES

__all__ = ["NavigationMap2D"]


@TRANSFORMABLES.register_module()
class NavigationMap2D(SpatialTransformable):
    def __init__(
        self,
        name,
        occupancy,
        clearance,
        traversability,
        source_pixel_to_body,
        tensor_smith=None,
    ):
        super().__init__(name)
        self.occupancy = np.asarray(occupancy).copy()
        self.clearance = np.asarray(clearance).copy()
        self.traversability = np.asarray(traversability).copy()
        if (
            self.occupancy.ndim != 2
            or self.clearance.shape != self.occupancy.shape
            or self.traversability.shape != self.occupancy.shape
        ):
            raise ValueError("navigation map rasters must have matching 2D shapes")
        if not np.isin(self.occupancy, (0, 127, 255)).all():
            raise ValueError("occupancy must contain only 0, 127, and 255")
        if (
            self.clearance.dtype.kind != "f"
            or not np.isfinite(self.clearance).all()
            or (self.clearance < 0).any()
        ):
            raise ValueError("clearance must be finite, nonnegative floating meters")
        if self.traversability.dtype != np.bool_:
            raise ValueError("traversability must be bool")
        self.source_pixel_to_body = np.asarray(
            source_pixel_to_body, dtype=np.float64
        ).copy()
        if self.source_pixel_to_body.shape != (3, 3):
            raise ValueError("source_pixel_to_body must be 3x3")
        for value in (
            self.occupancy,
            self.clearance,
            self.traversability,
            self.source_pixel_to_body,
        ):
            value.setflags(write=False)
        self.tensor_smith = tensor_smith

    def flip_3d(self, flip_mat, **kwargs):
        flip_mat = np.asarray(flip_mat, dtype=np.float64)
        planar = (
            flip_mat.shape == (3, 3)
            and np.allclose(flip_mat[2, :2], 0)
            and np.allclose(flip_mat[:2, 2], 0)
        )
        diagonal = planar and np.allclose(
            flip_mat[:2, :2], np.diag(np.diag(flip_mat[:2, :2]))
        )
        if (
            not diagonal
            or flip_mat[2, 2] != 1
            or not np.all(np.isin(np.diag(flip_mat)[:2], (-1, 1)))
        ):
            raise ValueError("NavigationMap2D only supports planar reflections")
        self.source_pixel_to_body = flip_mat @ self.source_pixel_to_body
        return self

    def rotate_3d(self, rmat, **kwargs):
        rmat = np.asarray(rmat, dtype=np.float64)
        planar = (
            rmat.shape == (3, 3)
            and np.allclose(rmat[2, :2], 0)
            and np.allclose(rmat[:2, 2], 0)
        )
        proper_yaw = (
            planar
            and np.isclose(rmat[2, 2], 1)
            and np.allclose(rmat[:2, :2].T @ rmat[:2, :2], np.eye(2))
            and np.isclose(np.linalg.det(rmat[:2, :2]), 1)
        )
        if not proper_yaw:
            raise ValueError("NavigationMap2D only supports planar yaw rotations")
        self.source_pixel_to_body = rmat @ self.source_pixel_to_body
        return self

    def translate_3d(self, translation, **kwargs):
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        left = np.eye(3)
        left[:2, 2] = -translation[:2]
        self.source_pixel_to_body = left @ self.source_pixel_to_body
        return self
