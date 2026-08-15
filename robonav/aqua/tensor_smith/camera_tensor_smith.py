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
        tensor_dict = dict(
            img=torch.tensor(np.float32(transformable.img / self.max_depth)),
            ego_mask=torch.tensor(transformable.ego_mask),
        )
        return tensor_dict
