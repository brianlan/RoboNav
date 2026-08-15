import torch

from prefusion import BaseModel

from robonav.registry import MODELS

__all__ = ["AquaNet"]


@MODELS.register_module()
class AquaNet(BaseModel):
    def __init__(
        self,
        *,
        data_preprocessor=None,
        backbone=None,
        attention_and_fusion=None,
        depth_head=None,
        trajectory_head=None,
        **kwargs,
    ):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.attention_and_fusion = MODELS.build(attention_and_fusion)
        self.depth_head = MODELS.build(depth_head)
        self.trajectory_head = MODELS.build(trajectory_head)

    def forward(
        self,
        *,
        index_info=None,
        camera_images=None,
        camera_depths=None,
        position_embedding=None,
        ego_poses=None,
        goal=None,
        future_trajectory=None,
        mode="loss",
        **kwargs,
    ):
        B = len(camera_images)
        camera_images = torch.row_stack(camera_images)
        pe = torch.row_stack(position_embedding)
        f1, f2, f3, f4, final_feat, hidden = self.backbone(
            camera_images, pe, goal, ego_poses
        )
        if mode == "loss":
            # TODO: dummy loss, replace with real head
            return dict(loss=sum(f.mean() for f in f4.values()))
        return f4
