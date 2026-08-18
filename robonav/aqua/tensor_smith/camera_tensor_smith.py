import torch
import numpy as np
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.transform import CameraDepth

from robonav.registry import TENSOR_SMITHS


__all__ = ["CameraDepthTensor"]


@TENSOR_SMITHS.register_module()
class CameraDepthTensor(TensorSmith):
    def __init__(self, *, max_depth=None):
        super().__init__()
        assert max_depth is not None, "max_depth is not provided."
        self.max_depth = max_depth

    def __call__(self, transformable: CameraDepth):
        if transformable.depth_mode != "d":
            raise ValueError(
                "CameraDepthTensor requires depth_mode='d' for Euclidean ray distance, "
                f"got {transformable.depth_mode!r}"
            )
        depth = np.asarray(transformable.img, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[..., 0]
        elif depth.ndim != 2:
            raise ValueError(f"Camera depth must have shape H,W or H,W,1, got {depth.shape}")
        ego_valid = np.asarray(transformable.ego_mask, dtype=bool)
        valid_mask = np.isfinite(depth) & (depth > 0) & ego_valid
        normalized = np.where(valid_mask, depth / self.max_depth, 0)
        tensor_dict = dict(
            img=torch.from_numpy(np.clip(normalized, 0, 1)[None]),
            valid_mask=torch.from_numpy(valid_mask[None]),
            ego_mask=torch.tensor(transformable.ego_mask),
        )
        return tensor_dict
