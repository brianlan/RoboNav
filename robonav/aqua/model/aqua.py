import torch
from scipy.spatial.transform import Rotation

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

    @staticmethod
    def _get_cur_velocity(ego_poses):
        vx_vy = torch.vstack([e["0"]["linear_velocity"].flatten()[:2] for e in ego_poses])
        omega = torch.vstack([e["0"]["angular_velocity"].flatten()[2] for e in ego_poses])
        return torch.concat([vx_vy, omega], dim=1)

    @staticmethod
    def _calc_delta_pose(ego_poses, device):
        if all(["-1" not in e for e in ego_poses]):
            return torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
        delta_poses = []
        for e in ego_poses:
            R_w_e1, t_w_e1 = e["-1"]["rotation"], e["-1"]["translation"]
            R_w_e2, t_w_e2 = e["0"]["rotation"], e["0"]["translation"]

            # R_e1_e2 = R_e1_w @ R_w_e2 = R'_w_e1 @ R_w_e2
            R_e1_e2 = R_w_e1.T @ R_w_e2
            delta_yaw = Rotation.from_matrix(R_e1_e2.detach().cpu().numpy()).as_euler("XYZ", degrees=False)[2]
            delta_yaw = torch.tensor(delta_yaw, device=R_e1_e2.device, dtype=R_e1_e2.dtype)

            # t_e1_e2 = R_e1_w @ t_w_e2 + t_e1_w = R'_w_e1 @ t_w_e2 - R'_w_e1 @ t_w_e1 = R'_w_e1 @ (t_w_e2 - t_w_e1)
            t_e1_e2 = R_w_e1.T @ (t_w_e2 - t_w_e1)
            delta_pose = torch.cat([t_e1_e2.flatten()[:2], delta_yaw.unsqueeze(0)])
            delta_poses.append(delta_pose)

        return torch.vstack(delta_poses)
