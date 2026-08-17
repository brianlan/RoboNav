import torch.nn as nn
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS


__all__ = ["DepthHead"]


@MODELS.register_module()
class DepthHead(BaseModel):
    def __init__(self, f4_chans, f3_chans, f2_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f4_conv_1x1 = nn.Conv2d(f4_chans, f2_chans, 1)
        self.f4_conv_3x3 = nn.Conv2d(f2_chans, f2_chans, 3, padding=1)
        self.f3_conv_1x1 = nn.Conv2d(f3_chans, f2_chans, 1)
        self.f3_conv_3x3 = nn.Conv2d(f2_chans, f2_chans, 3, padding=1)
        self.f2_conv_1x1 = nn.Conv2d(f2_chans, f2_chans, 1)
        self.f2_conv_3x3 = nn.Conv2d(f2_chans, f2_chans, 3, padding=1)
        self.f2_up_conv_3x3 = nn.Conv2d(f2_chans, f2_chans, 3, padding=1)

    def forward(self, f4, f3, f2):
        f4d = self.f4_conv_1x1(f4)
        f3d = self.f3_conv_1x1(f3) + F.interpolate(f4d, scale_factor=2, mode="nearest")
        f2d = self.f2_conv_1x1(f2) + F.interpolate(f3d, scale_factor=2, mode="nearest")
        f2d_up = F.interpolate(f2d, scale_factor=2, mode="nearest")
        f4d = self.f4_conv_3x3(f4d)
        f3d = self.f3_conv_3x3(f3d)
        f2d = self.f2_conv_3x3(f2d)
        f2d_up = self.f2_up_conv_3x3(f2d_up)
        return f4d, f3d, f2d, f2d_up
