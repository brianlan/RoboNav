import torch.nn as nn

from prefusion import BaseModel
from robonav.aqua.registry import MODELS


@MODELS.register_module()
class AquaNet(BaseModel):
    def __init__(self, *, backbone=None, neck=None, **kwargs):
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

    def forward(self):
        pass
