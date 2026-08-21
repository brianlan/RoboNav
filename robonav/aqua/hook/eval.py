import torch
from mmengine.hooks import Hook

from robonav.aqua.metric import frame_trajectory_metrics
from robonav.registry import HOOKS

__all__ = ["AquaTrajectoryEvalHook"]


@HOOKS.register_module()
class AquaTrajectoryEvalHook(Hook):
    """Streaming trajectory evaluation for AquaNet test runs.

    ``SequenceBatchInferLoop`` never calls an evaluator, so this hook
    accumulates scalar per-frame metrics on CPU and reports epoch
    aggregates. Recurrent state is the model's concern: AquaNet resets
    itself on the ``stream_start`` the data pipeline marks at scene
    starts. Predictions are the mode="tensor" outputs (B, K, 7 layout
    [x, y, sin(yaw), cos(yaw), vx, vy, omega]); ground truth comes from
    ``data_batch[0]["future_trajectory"]``. Metric names carry their
    units (m, rad, mps = m/s, radps = rad/s).
    """

    def __init__(self, delta_t=0.1):
        self.delta_t = delta_t
        self._sums = {}
        self._frames = 0

    def before_test(self, runner):
        self._sums = {}
        self._frames = 0

    def after_test_iter(
        self, runner, batch_idx, data_batch=None, outputs=None, mode="test"
    ):
        metrics = self._evaluate_frame(data_batch, outputs)
        self._record_metrics(metrics)
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

    def _evaluate_frame(self, data_batch, outputs):
        sample = data_batch[0]
        pred7 = outputs.detach().to("cpu", torch.float32)
        gt7 = sample["future_trajectory"].to("cpu", torch.float32).unsqueeze(0)
        twist = sample.get("twist")
        if torch.is_tensor(twist):
            twist = twist.to("cpu", torch.float32)
        return self._frame_metrics(pred7, gt7, twist)

    def _record_metrics(self, metrics):
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + value

    def _frame_metrics(self, pred7, gt7, twist):
        if torch.is_tensor(twist):
            twist = twist.unsqueeze(0)
        return frame_trajectory_metrics(pred7, gt7, twist, self.delta_t)
