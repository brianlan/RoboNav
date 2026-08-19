import math

import torch
from mmengine.evaluator import BaseMetric

from robonav.aqua.tensor_smith.future_trajectory_tensor_smith import (
    FutureTrajectoryTensorSmith,
)
from robonav.registry import METRICS

__all__ = ["frame_trajectory_metrics", "AquaTrajectoryMetric"]

_smith = FutureTrajectoryTensorSmith()


def _wrap_angle(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _trapezoid(values, delta_t):
    return delta_t * (values[:, 1:] + values[:, :-1]) / 2


def _kinematic_errors(pred, twist, delta_t):
    """Trapezoidal integration of the predicted velocities, seeded
    with the current twist, must reproduce the predicted poses."""
    xy = pred[..., :2]
    yaw = pred[..., 2]
    prev_xy = torch.cat((torch.zeros_like(xy[:, :1]), xy[:, :-1]), dim=1)
    vel = torch.cat((twist[:, :2].unsqueeze(1), pred[..., 3:5]), dim=1)
    pos_residual = (xy - prev_xy) - _trapezoid(vel, delta_t)
    prev_yaw = torch.cat((torch.zeros_like(yaw[:, :1]), yaw[:, :-1]), dim=1)
    omega = torch.cat((twist[:, 2:3], pred[..., 5]), dim=1)
    yaw_residual = _wrap_angle((yaw - prev_yaw) - _trapezoid(omega, delta_t))
    return {
        "kin_pos_err_m": pos_residual.norm(dim=-1).mean().item(),
        "kin_yaw_err_rad": yaw_residual.abs().mean().item(),
    }


def frame_trajectory_metrics(pred7, gt7, twist, delta_t):
    """Pure per-frame trajectory metrics, the single calculation shared
    by the test hook and the validation metric.

    ``pred7``/``gt7`` are (B, K, 7) state tensors laid out as
    [x, y, sin(yaw), cos(yaw), vx, vy, omega]; ``twist`` is (B, 3) or
    None (None skips the kinematic checks). Returns a dict of scalar
    means over the batch. Metric names carry their units (m, rad,
    mps = m/s, radps = rad/s).
    """
    pred, metrics = _trajectory_metrics(pred7, gt7)
    if twist is not None:
        metrics.update(_kinematic_errors(pred, twist, delta_t))
    return metrics


def _trajectory_metrics(pred7, gt7):
    pred = _smith.reverse(pred7)  # (B, K, 6): x, y, yaw, vx, vy, omega
    gt = _smith.reverse(gt7)
    xy_err = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
    yaw_err = _wrap_angle(pred[..., 2] - gt[..., 2]).abs()
    lin_vel_err = (pred[..., 3:5] - gt[..., 3:5]).norm(dim=-1)
    ang_vel_err = (pred[..., 5] - gt[..., 5]).abs()
    return pred, {
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

@METRICS.register_module()
class AquaTrajectoryMetric(BaseMetric):
    """MMEngine validation adapter around ``frame_trajectory_metrics``.

    Processes one per-sample prediction (``pred_trajectory``, (K, 7))
    against the ground truth from the data batch; ``compute_metrics``
    averages every accumulated per-sample dict.
    """

    default_prefix = "trajectory"

    def __init__(self, delta_t=0.1, **kwargs):
        super().__init__(**kwargs)
        self.delta_t = delta_t

    def process(self, data_batch, data_samples):
        for data, sample in zip(data_batch, data_samples):
            self.results.append(self._evaluate_sample(data, sample))

    def _evaluate_sample(self, data, sample):
        pred7 = sample["pred_trajectory"].to(torch.float32).unsqueeze(0)
        gt7 = data["future_trajectory"].to(torch.float32).unsqueeze(0)
        twist = data.get("twist")
        if torch.is_tensor(twist):
            twist = twist.to(torch.float32).unsqueeze(0)
        return frame_trajectory_metrics(pred7, gt7, twist, self.delta_t)

    def compute_metrics(self, results):
        return {
            name: sum(result[name] for result in results) / len(results)
            for name in results[0]
        }

