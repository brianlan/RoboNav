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
        feature_modulation=None,
        temporal_fuser=None,
        depth_head=None,
        trajectory_head=None,
        **kwargs,
    ):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.feature_modulation = MODELS.build(feature_modulation)
        self.temporal_fuser = MODELS.build(temporal_fuser)
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
        delta_poses=None,
        twist=None,
        goal=None,
        future_trajectory=None,
        mode="loss",
        **kwargs,
    ):
        B = len(camera_images)
        camera_images = torch.row_stack(camera_images)
        device = camera_images.device
        pe = torch.row_stack(position_embedding)
        f1, f2, f3, f4 = self.backbone(camera_images, pe)

        cur_velo = self._get_cur_velocity(ego_poses)
        delta_pose = self._calc_delta_pose(ego_poses, device)

        # f3g = self.feature_modulation(f4, f3, cur_velo, goal)
        # final_feat, hidden = self.temporal_fuser(f3g, cur_velo, delta_pose, goal)

        # f4d, f3d, f2d, f2d_up = self.depth_head(f4, f3, f2)

        # trajectory = self.trajectory_head(final_feat, hidden)

        if mode == "loss":
            # TODO: dummy loss, replace with real head
            return dict(loss=sum(f.mean() for f in f4))
        return f4
