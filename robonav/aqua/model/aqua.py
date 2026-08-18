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

    def reset(self):
        self.temporal_fuser.reset()

    def train_step(self, data, optim_wrapper):
        """Full-sequence BPTT: ``data`` is one sequence of aligned frame
        microbatches. Forward every frame while retaining the graph, parse
        each frame's loss dict, average over time, then call backward and
        step the optimizer once."""
        with optim_wrapper.optim_context(self):
            sequence = [self.data_preprocessor(frame, True) for frame in data]
            self.reset()
            parsed = [
                self.parse_losses(self._run_forward(frame, mode="loss"))
                for frame in sequence
            ]
            loss = torch.stack([scalar for scalar, _ in parsed]).mean()
        log_vars = {
            key: sum(frame_vars[key] for _, frame_vars in parsed) / len(parsed)
            for key in parsed[0][1]
        }
        # parse_losses overwrites its inserted 'loss' entry when the frame
        # dict has a literal 'loss' key; report the actual optimized scalar.
        log_vars["loss"] = loss
        optim_wrapper.update_params(loss)
        self.reset()
        return log_vars

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
        pe = torch.row_stack(position_embedding)
        goal = torch.row_stack(goal)
        twist = torch.row_stack(twist)
        delta_poses = torch.row_stack(delta_poses)

        f1, f2, f3, f4 = self.backbone(camera_images, pe)
        f3g = self.feature_modulation(f4, f3, twist, goal)
        final_feat, hidden = self.temporal_fuser(f3g, twist, delta_poses, goal)
        f4d, f3d, f2d, f2d_up = self.depth_head(f4, f3, f2)
        trajectory = self.trajectory_head(final_feat, hidden)

        if mode == "loss":
            # TODO: dummy loss, replace with real head
            return dict(loss=sum(f.mean() for f in f4))
        return f4
