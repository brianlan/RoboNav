import torch
from collections import OrderedDict

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
        self.hidden = None
        self.state = None

    def forward(
        self,
        *,
        index_info=None,
        camera_images=None,
        camera_depths=None,
        position_embedding=None,
        ego_poses=None,
        goals=None,
        mode="loss",
        **kwargs,
    ):
        B = len(camera_images)
        camera_images = torch.row_stack(camera_images)
        pe = torch.row_stack(position_embedding)
        # goal = torch.row_stack(goals)
        f1, f2, f3, f4, self.hidden = self.backbone(camera_images, pe, None, self.state, self.hidden)
        feats = self.neck(OrderedDict(zip(["0", "1", "2", "3"], [f1, f2, f3, f4])))
        if mode == "loss":
            # TODO: dummy loss, replace with real head
            return dict(loss=sum(f.mean() for f in feats.values()))
        return feats
