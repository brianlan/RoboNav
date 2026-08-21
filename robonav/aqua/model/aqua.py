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

    def forward(self, *, sequence=None, mode="loss", **frame):
        if sequence is not None:
            return self._forward_sequence(sequence, mode)
        return self._forward_frame(mode=mode, **frame)

    def _forward_sequence(self, sequence, mode):
        """Full-sequence BPTT behind the default ``train_step`` (single- and
        multi-GPU): forward every frame in order while retaining the graph,
        then average every loss and metric key over time. Resetting at both
        ends keeps the recurrent state sequence-local; the surrounding
        ``train_step`` parses the averaged dict and performs exactly one
        backward and one optimizer step per sequence."""
        self.reset()
        frame_outputs = [self._forward_frame(mode=mode, **frame) for frame in sequence]
        self.reset()
        return {
            key: torch.stack([output[key] for output in frame_outputs]).mean()
            for key in frame_outputs[0]
        }

    def _forward_frame(
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
        camera_images, position_embedding, twist, delta_poses, goal = self._stack_inputs(
            camera_images, position_embedding, twist, delta_poses, goal
        )
        f4, depth_predictions, trajectory = self._predict_outputs(
            camera_images, position_embedding, twist, delta_poses, goal
        )

        if mode in ("loss", "predict"):
            losses = self._compute_losses(
                trajectory,
                twist,
                camera_depths,
                camera_depth_valid_masks,
                occupancy,
                clearance,
                traversability,
                future_trajectory,
                depth_predictions,
            )
            if mode == "loss":
                return losses
            return self._format_predictions(trajectory, losses)
        if mode == "tensor":
            return trajectory
        return f4

    def _stack_inputs(self, camera_images, position_embedding, twist, delta_poses, goal):
        return (
            torch.row_stack(camera_images),
            torch.row_stack(position_embedding),
            torch.row_stack(twist),
            torch.row_stack(delta_poses),
            torch.row_stack(goal),
        )

    def _predict_outputs(
        self, camera_images, position_embedding, twist, delta_poses, goal
    ):
        f1, f2, f3, f4 = self.backbone(camera_images, position_embedding)
        f3g = self.feature_modulation(f4, f3, twist, goal)
        final_feat, hidden = self.temporal_fuser(f3g, twist, delta_poses, goal)
        depth_predictions = self.depth_head(f4, f3, f2, f1)
        trajectory = self.trajectory_head(final_feat, hidden)
        return f4, depth_predictions, trajectory

    def _compute_losses(
        self,
        trajectory,
        twist,
        camera_depths,
        camera_depth_valid_masks,
        occupancy,
        clearance,
        traversability,
        future_trajectory,
        depth_predictions,
    ):
        camera_depths = torch.row_stack(camera_depths)
        camera_depth_valid_masks = torch.row_stack(camera_depth_valid_masks)
        occupancy = torch.row_stack(occupancy)
        clearance = torch.row_stack(clearance)
        traversability = torch.row_stack(traversability)
        # list elements are per-sample (K, 7); stack keeps (B, K, 7)
        future_trajectory = torch.stack(future_trajectory)
        return self.loss_module(
            trajectory=trajectory,
            trajectory_target=future_trajectory,
            twist=twist,
            depth_predictions=depth_predictions,
            depth_target=camera_depths,
            depth_valid_mask=camera_depth_valid_masks,
        )

    def _format_predictions(self, trajectory, losses):
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
