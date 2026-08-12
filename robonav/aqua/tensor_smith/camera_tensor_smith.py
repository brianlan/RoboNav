from typing import Iterable

import torch
import numpy as np
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.transform import CameraImage, CameraDepth

from robonav.registry import TENSOR_SMITHS


__all__ = ["CameraImageTensor", "CameraDepthTensor"]


@TENSOR_SMITHS.register_module()
class CameraImageTensor(TensorSmith):
    def __init__(
        self,
        means: list[float, float, float] | tuple[float, float, float] | float = 128,
        stds: list[float, float, float] | tuple[float, float, float] | float = 255,
    ):
        if isinstance(means, Iterable):
            means = np.array(means, dtype=np.float32)
        if isinstance(stds, Iterable):
            stds = np.array(stds, dtype=np.float32)
        self.means = means
        self.stds = stds

    def __call__(self, transformable: CameraImage):
        tensor_dict = dict(
            img=torch.tensor(
                np.float32((transformable.img - self.means) / self.stds).transpose(
                    2, 0, 1
                )
            ),
            ego_mask=torch.tensor(transformable.ego_mask),
        )
        return tensor_dict


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
