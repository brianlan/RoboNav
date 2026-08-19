import math

import torch
from mmengine.hooks import Hook

from robonav.aqua.tensor_smith.future_trajectory_tensor_smith import (
    FutureTrajectoryTensorSmith,
)
from robonav.registry import HOOKS

__all__ = ["AquaTrajectoryEvalHook"]


def _wrap_angle(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _unwrap_model(runner):
    model = runner.model
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class AquaTrajectoryEvalHook(Hook):
    """Streaming trajectory evaluation for AquaNet test runs.

    ``SequenceBatchInferLoop`` never calls an evaluator, so this hook
    resets the recurrent model at scene boundaries, accumulates scalar
    per-frame metrics on CPU, and reports epoch aggregates. Predictions
    are the mode="tensor" outputs (B, K, 7 layout
    [x, y, sin(yaw), cos(yaw), vx, vy, omega]); ground truth comes from
    ``data_batch[0]["future_trajectory"]``. Metric names carry their
    units (m, rad, mps = m/s, radps = rad/s).
    """

    def __init__(self, delta_t=0.1):
        self.delta_t = delta_t
        self.smith = FutureTrajectoryTensorSmith()
        self._scene_id = None
        self._sums = {}
        self._frames = 0

    def before_test(self, runner):
        self._scene_id = None
        self._sums = {}
        self._frames = 0
        _unwrap_model(runner).reset()

    def before_test_iter(self, runner, batch_idx, data_batch):
        scene_id = data_batch[0]["index_info"].scene_id
        if scene_id != self._scene_id:
            _unwrap_model(runner).reset()
            self._scene_id = scene_id

    def after_test_iter(
        self, runner, batch_idx, data_batch=None, outputs=None, mode="test"
    ):
        pred7 = outputs.detach().to("cpu", torch.float32)
        gt7 = data_batch[0]["future_trajectory"].to("cpu", torch.float32).unsqueeze(0)
        twist = data_batch[0].get("twist")
        if torch.is_tensor(twist):
            twist = twist.to("cpu", torch.float32)
        for name, value in self._frame_metrics(pred7, gt7, twist).items():
            self._sums[name] = self._sums.get(name, 0.0) + value
        self._frames += 1

    def after_test_epoch(self, runner, metrics=None):
        if not self._frames:
            return
        names = sorted(self._sums)
        width = max(len(name) for name in names)
        table = "\n".join(
            f"{name:<{width}}  {self._sums[name] / self._frames:.4f}" for name in names
        )
        runner.logger.info(
            f"Aqua trajectory evaluation over {self._frames} frames:\n{table}"
        )

    def _frame_metrics(self, pred7, gt7, twist):
        pred = self.smith.reverse(pred7)  # (1, K, 6): x, y, yaw, vx, vy, omega
        gt = self.smith.reverse(gt7)
        xy_err = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
        yaw_err = _wrap_angle(pred[..., 2] - gt[..., 2]).abs()
        lin_vel_err = (pred[..., 3:5] - gt[..., 3:5]).norm(dim=-1)
        ang_vel_err = (pred[..., 5] - gt[..., 5]).abs()
        metrics = {
            "ADE_m": xy_err.mean().item(),
            "FDE_m": xy_err[:, -1].mean().item(),
            "yaw_mae_rad": yaw_err.mean().item(),
            "terminal_yaw_err_rad": yaw_err[:, -1].mean().item(),
            "lin_vel_err_mps": lin_vel_err.mean().item(),
            "ang_vel_err_radps": ang_vel_err.mean().item(),
            "terminal_lin_vel_err_mps": lin_vel_err[:, -1].mean().item(),
            "terminal_ang_vel_err_radps": ang_vel_err[:, -1].mean().item(),
            "unit_circle_err": (
                (pred7[..., 2] ** 2 + pred7[..., 3] ** 2 - 1).abs().mean().item()
            ),
        }
        if twist is not None:
            metrics.update(self._kinematic_errors(pred, twist))
        return metrics

    def _kinematic_errors(self, pred, twist):
        """Trapezoidal integration of the predicted velocities, seeded
        with the current twist, must reproduce the predicted poses."""
        xy = pred[..., :2]
        yaw = pred[..., 2]
        prev_xy = torch.cat((torch.zeros_like(xy[:, :1]), xy[:, :-1]), dim=1)
        vel = torch.cat((twist[:2].reshape(1, 1, 2), pred[..., 3:5]), dim=1)
        pos_residual = (xy - prev_xy) - self._trapezoid(vel)
        prev_yaw = torch.cat((torch.zeros_like(yaw[:, :1]), yaw[:, :-1]), dim=1)
        omega = torch.cat((twist[2].reshape(1, 1), pred[..., 5]), dim=1)
        yaw_residual = _wrap_angle((yaw - prev_yaw) - self._trapezoid(omega))
        return {
            "kin_pos_err_m": pos_residual.norm(dim=-1).mean().item(),
            "kin_yaw_err_rad": yaw_residual.abs().mean().item(),
        }

    def _trapezoid(self, values):
        return self.delta_t * (values[:, 1:] + values[:, :-1]) / 2
