import torch
import torch.nn as nn

from prefusion import BaseModel

from robonav.registry import MODELS

__all__ = ["AquaNet"]


@MODELS.register_module()
class AquaNet(BaseModel):
    def __init__(self, *, data_preprocessor=None, backbone=None, neck=None, **kwargs):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

    def forward(
        self,
        *,
        index_info=None,
        cameras=None,
        depths=None,
        ego_poses=None,
        mode="loss",
        **kwargs,
    ):
        feat = self.backbone(cameras)
