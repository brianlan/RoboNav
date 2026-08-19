import math

import torch
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["AquaLoss"]

# Trajectory state layout: [x, y, sin(yaw), cos(yaw), vx, vy, omega]
# (docs/aquanet_loss_supervision_plan.md §2.1). Phase 1 contract: §4, §6.
_TERM_NAMES = (
    "traj_xy",
    "traj_yaw",
    "traj_unit",
    "traj_vel",
    "kin_pos",
    "kin_yaw",
    "depth",
)


@MODELS.register_module()
class AquaLoss(BaseModel):
    """Phase 1 loss for AquaNet.

    Returns one MMEngine loss dict: weighted trainable terms under ``loss_*``
    keys (summed exactly once by ``parse_losses``) plus detached raw values
    and metrics whose names never contain ``loss``. Velocities are already in
    the current-frame body frame, so kinematic consistency never applies
    R(yaw) to them.
    """

    def __init__(
        self,
        *,
        loss_weights,
        depth_loss,
        delta_t=0.1,
        terminal_weight=1.0,
        xy_scale=1.0,
        vel_scale=1.0,
        omega_scale=1.0,
        v_stop_threshold=0.1,
        omega_stop_threshold=0.1,
        eps=1e-6,
        beta_xy=1.0,
        beta_vel=1.0,
        beta_kin_pos=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if set(loss_weights) != set(_TERM_NAMES):
            raise ValueError(
                f"loss_weights keys must be {_TERM_NAMES}, got {sorted(loss_weights)}"
            )
        self.loss_weights = dict(loss_weights)
        self.depth_loss = MODELS.build(depth_loss)
        self.delta_t = delta_t
        self.terminal_weight = terminal_weight
        self.xy_scale = xy_scale
        self.vel_scale = vel_scale
        self.omega_scale = omega_scale
        self.v_stop_threshold = v_stop_threshold
        self.omega_stop_threshold = omega_stop_threshold
        self.eps = eps
        self.beta_xy = beta_xy
        self.beta_vel = beta_vel
        self.beta_kin_pos = beta_kin_pos

    def forward(
        self,
        *,
        trajectory,
        trajectory_target,
        twist,
        depth_predictions,
        depth_target,
        depth_valid_mask,
    ):
        if (
            trajectory.ndim != 3
            or trajectory.shape[-1] != 7
            or trajectory.shape != trajectory_target.shape
        ):
            raise ValueError(
                "trajectory and trajectory_target must have identical "
                f"B,K,7 shapes, got {tuple(trajectory.shape)} "
                f"and {tuple(trajectory_target.shape)}"
            )
        if tuple(twist.shape) != (trajectory.shape[0], 3):
            raise ValueError(
                f"twist must have shape B,3 with B={trajectory.shape[0]}, "
                f"got {tuple(twist.shape)}"
            )
        twist_scales = trajectory.new_tensor(
            [self.vel_scale, self.vel_scale, self.omega_scale]
        )
        traj_terms = self._trajectory_terms(
            trajectory, trajectory_target, twist_scales
        )
        kin_pos, kin_yaw = self._kinematic_consistency(trajectory, twist)
        depth = self.depth_loss(depth_predictions, depth_target, depth_valid_mask)

        raw = dict(traj_terms, kin_pos=kin_pos, kin_yaw=kin_yaw, depth=depth)
        losses = {}
        for name, value in raw.items():
            losses[f"loss_{name}"] = value * self.loss_weights[name]
            losses[f"{name}_raw"] = value.detach()
        losses.update(self._metrics(trajectory, trajectory_target, twist_scales))
        return losses

    def _trajectory_terms(self, trajectory, trajectory_target, twist_scales):
        """Per-state supervision: terminal-weighted xy error, yaw direction
        error, unit-norm penalty, and scaled velocity error."""
        pred_xy = trajectory[..., :2]
        pred_q = trajectory[..., 2:4]
        target_xy = trajectory_target[..., :2]
        target_q = trajectory_target[..., 2:4]

        time_weight = trajectory.new_ones(trajectory.shape[0], trajectory.shape[1], 1)
        time_weight[:, -1] = self.terminal_weight
        traj_xy = self._weighted_mean(
            self._smooth_l1((pred_xy - target_xy) / self.xy_scale, self.beta_xy),
            time_weight,
        )

        pred_norm = pred_q.norm(dim=-1)
        traj_yaw = (
            1 - (pred_q * target_q).sum(-1) / pred_norm.clamp_min(self.eps)
        ).mean()
        traj_unit = ((pred_norm**2 - 1) ** 2).mean()

        traj_vel = self._smooth_l1(
            (trajectory[..., 4:] - trajectory_target[..., 4:]) / twist_scales,
            self.beta_vel,
        ).mean()
        return dict(
            traj_xy=traj_xy, traj_yaw=traj_yaw, traj_unit=traj_unit, traj_vel=traj_vel
        )

    def _kinematic_consistency(self, trajectory, twist):
        """From the origin pose and current twist, trapezoidal integration
        of the predicted velocities must reproduce the predicted positions
        and yaw angles."""
        pred_xy = trajectory[..., :2]
        pred_q = trajectory[..., 2:4]
        batch = pred_xy.shape[0]

        position = torch.cat((pred_xy.new_zeros(batch, 1, 2), pred_xy), dim=1)
        velocity = torch.cat((twist[:, :2].unsqueeze(1), trajectory[..., 4:6]), dim=1)
        position_residual = (
            position[:, 1:] - position[:, :-1] - self._trapezoid(velocity)
        )
        kin_pos = self._smooth_l1(
            position_residual / self.xy_scale, self.beta_kin_pos
        ).mean()

        sin = torch.cat((pred_q.new_zeros(batch, 1, 1), pred_q[..., 0:1]), dim=1)
        cos = torch.cat((pred_q.new_ones(batch, 1, 1), pred_q[..., 1:2]), dim=1)
        norm = (sin**2 + cos**2).sqrt().clamp_min(self.eps)
        sin, cos = sin / norm, cos / norm
        sin_delta = sin[:, 1:] * cos[:, :-1] - cos[:, 1:] * sin[:, :-1]
        cos_delta = cos[:, 1:] * cos[:, :-1] + sin[:, 1:] * sin[:, :-1]
        omega = torch.cat((twist[:, 2:3].unsqueeze(1), trajectory[..., 6:7]), dim=1)
        omega_delta = self._trapezoid(omega)
        kin_yaw = (
            1 - sin_delta * omega_delta.sin() - cos_delta * omega_delta.cos()
        ).mean()
        return kin_pos, kin_yaw

    def _trapezoid(self, values):
        """Per-step trapezoidal integral: delta_t * (v_k + v_{k+1}) / 2."""
        return self.delta_t * (values[:, 1:] + values[:, :-1]) / 2

    @torch.no_grad()
    def _metrics(self, trajectory, trajectory_target, vel_scales):
        pred = trajectory.detach()
        target = trajectory_target.detach()
        # scale-normalized like the loss so m/s and rad/s combine into a
        # dimensionless diagnostic
        velocity_error = (
            (pred[..., 4:] - target[..., 4:]).abs() / vel_scales
        ).mean(dim=-1)
        stopped = (target[..., 4:6].norm(dim=-1) < self.v_stop_threshold) & (
            target[..., 6].abs() < self.omega_stop_threshold
        )
        moving = ~stopped
        yaw_delta = (
            torch.atan2(pred[..., 2], pred[..., 3])
            - torch.atan2(target[..., 2], target[..., 3])
            + math.pi
        ) % (2 * math.pi) - math.pi
        return dict(
            velocity_error_moving=self._weighted_mean(
                velocity_error, moving.float()
            ),
            velocity_error_stopped=self._weighted_mean(
                velocity_error, stopped.float()
            ),
            terminal_velocity_error=velocity_error[:, -1].mean(),
            stop_sample_ratio=stopped.float().mean(),
            ADE=(pred[..., :2] - target[..., :2]).norm(dim=-1).mean(),
            FDE=(pred[:, -1, :2] - target[:, -1, :2]).norm(dim=-1).mean(),
            yaw_error=yaw_delta.abs().mean(),
        )

    def _weighted_mean(self, values, weight):
        weight = weight.to(values.dtype).broadcast_to(values.shape)
        return (values * weight).sum() / weight.sum().clamp_min(self.eps)

    @staticmethod
    def _smooth_l1(residual, beta):
        return F.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=beta, reduction="none"
        )
