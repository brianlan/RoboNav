import torch
from mmengine.structures import BaseDataElement
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
        loss=None,
        trajectory_head=None,
        **kwargs,
    ):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.feature_modulation = MODELS.build(feature_modulation)
        self.temporal_fuser = MODELS.build(temporal_fuser)
        self.depth_head = MODELS.build(depth_head)
        self.loss_module = MODELS.build(loss)
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
        camera_depth_valid_masks=None,
        position_embedding=None,
        ego_poses=None,
        delta_poses=None,
        twist=None,
        goal=None,
        future_trajectory=None,
        occupancy=None,
        clearance=None,
        traversability=None,
        mode="loss",
        **kwargs,
    ):
        camera_images = torch.row_stack(camera_images)
        pe = torch.row_stack(position_embedding)
        goal = torch.row_stack(goal)
        twist = torch.row_stack(twist)
        delta_poses = torch.row_stack(delta_poses)

        f1, f2, f3, f4 = self.backbone(camera_images, pe)
        f3g = self.feature_modulation(f4, f3, twist, goal)
        final_feat, hidden = self.temporal_fuser(f3g, twist, delta_poses, goal)
        depth_predictions = self.depth_head(f4, f3, f2, f1)
        trajectory = self.trajectory_head(final_feat, hidden)

        if mode in ("loss", "predict"):
            camera_depths = torch.row_stack(camera_depths)
            camera_depth_valid_masks = torch.row_stack(camera_depth_valid_masks)
            occupancy = torch.row_stack(occupancy)
            clearance = torch.row_stack(clearance)
            traversability = torch.row_stack(traversability)
            # list elements are per-sample (K, 7); stack keeps (B, K, 7)
            future_trajectory = torch.stack(future_trajectory)
            losses = self.loss_module(
                trajectory=trajectory,
                trajectory_target=future_trajectory,
                twist=twist,
                depth_predictions=depth_predictions,
                depth_target=camera_depths,
                depth_valid_mask=camera_depth_valid_masks,
            )
            if mode == "loss":
                return losses
        if mode == "predict":
            # MMEngine ValLoop contract: one BaseDataElement per sample,
            # plus a trailing element whose only field is `loss`. Only
            # weighted loss_* keys belong there, so raw metrics cannot be
            # double-counted by MMEngine loss parsing.
            predictions = []
            for sample_trajectory in trajectory:
                prediction = BaseDataElement()
                prediction.pred_trajectory = sample_trajectory.detach().to("cpu")
                predictions.append(prediction)
            loss_element = BaseDataElement()
            loss_element.loss = {
                name: value.detach()
                for name, value in losses.items()
                if name.startswith("loss_")
            }
            return predictions + [loss_element]
        if mode == "tensor":
            return trajectory
        return f4
