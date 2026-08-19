import torch
from mmengine.hooks import Hook

from robonav.aqua.metric import frame_trajectory_metrics
from robonav.registry import HOOKS

__all__ = ["AquaTrajectoryEvalHook"]


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
        if torch.is_tensor(twist):
            twist = twist.unsqueeze(0)
        return frame_trajectory_metrics(pred7, gt7, twist, self.delta_t)
