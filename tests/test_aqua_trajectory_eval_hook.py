import math
import os
import types

import pytest
import torch
from mmengine.config import Config

from robonav.aqua.hook.eval import AquaTrajectoryEvalHook
from robonav.aqua.model.aqua import AquaNet
from robonav.aqua.model.trajectory_head import TrajectoryHead

DELTA_T = 0.1


class _StubBackbone(torch.nn.Module):
    def forward(self, camera_images, position_embedding):
        feat = camera_images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        return feat, feat, feat, feat


class _StubFeatureModulation(torch.nn.Module):
    def forward(self, f4, f3, twist, goal):
        return f3


class _StubTemporalFuser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, f3g, twist, delta_poses, goal):
        batch = twist.shape[0]
        self.state = torch.zeros(batch)
        return torch.ones(batch, 8), torch.zeros(batch, 4)


class _StubDepthHead(torch.nn.Module):
    def forward(self, f4, f3, f2, f1):
        return f4


class StubAquaNet(AquaNet):
    """AquaNet with stub feature extractors but the real forward and a
    real TrajectoryHead, to test the mode="tensor" output contract."""

    def __init__(self, num_steps):
        torch.nn.Module.__init__(self)
        self.backbone = _StubBackbone()
        self.feature_modulation = _StubFeatureModulation()
        self.temporal_fuser = _StubTemporalFuser()
        self.depth_head = _StubDepthHead()
        self.loss_module = None
        self.trajectory_head = TrajectoryHead(
            feat_chans=8,
            hidden_chans=4,
            intermediate_chans=4,
            out_chans=7,
            num_trajectory_steps=num_steps,
        )


def _forward_inputs(batch):
    return dict(
        camera_images=[torch.randn(1, 3, 8, 8) for _ in range(batch)],
        position_embedding=[torch.randn(1, 6, 4, 4) for _ in range(batch)],
        goal=[torch.randn(6) for _ in range(batch)],
        twist=[torch.randn(3) for _ in range(batch)],
        delta_poses=[torch.zeros(3) for _ in range(batch)],
    )


def test_tensor_mode_returns_trajectory():
    num_steps = 5
    model = StubAquaNet(num_steps)
    for batch in (1, 2):
        outputs = model(**_forward_inputs(batch), mode="tensor")
        assert outputs.shape == (batch, num_steps, 7)
        assert torch.isfinite(outputs).all()


def test_predict_mode_returns_backbone_features():
    model = StubAquaNet(num_steps=5)
    batch = 2
    f4 = model(**_forward_inputs(batch), mode="predict")
    # stub backbone f4 is the (B, 1, 1, 1) pooled feature, not the
    # (B, K, 7) trajectory reserved for mode="tensor"
    assert f4.shape == (batch, 1, 1, 1)


class ResetCountingModel:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def _batch(scene_id):
    return [
        {
            "index_info": types.SimpleNamespace(scene_id=scene_id),
            "future_trajectory": torch.zeros(2, 7),
            "twist": torch.zeros(3),
        }
    ]


def test_resets_before_test_and_on_scene_transition():
    hook = AquaTrajectoryEvalHook()
    model = ResetCountingModel()
    runner = types.SimpleNamespace(model=model)

    hook.before_test(runner)
    assert model.reset_count == 1
    hook.before_test_iter(runner, 0, _batch("A"))
    assert model.reset_count == 2  # first frame of scene A
    hook.before_test_iter(runner, 1, _batch("A"))
    assert model.reset_count == 2  # same scene, no reset
    hook.before_test_iter(runner, 2, _batch("B"))
    assert model.reset_count == 3  # new scene


def _seven(xy, yaw, vel, omega, q_scale=1.0):
    x, y = xy
    vx, vy = vel
    return torch.tensor(
        [x, y, math.sin(yaw) * q_scale, math.cos(yaw) * q_scale, vx, vy, omega],
        dtype=torch.float32,
    )


def test_frame_metric_values_and_yaw_wrap():
    hook = AquaTrajectoryEvalHook()
    pred7 = torch.stack(
        [
            _seven((0.5, 0.0), 3.1, (1.0, 1.0), 0.4, q_scale=0.8),
            _seven((1.0, 1.0), 0.0, (0.0, 2.0), -0.6, q_scale=0.5),
        ]
    ).unsqueeze(0)
    gt7 = torch.stack(
        [_seven((0.0, 0.0), -3.1, (1.0, 0.0), 0.5), _seven((1.0, 0.0), 0.0, (0.0, 1.0), -0.5)]
    ).unsqueeze(0)

    metrics = hook._frame_metrics(pred7, gt7, None)

    wrapped_yaw_err = abs((6.2 + math.pi) % (2 * math.pi) - math.pi)
    assert metrics["ADE_m"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["FDE_m"] == pytest.approx(1.0)
    # float32 atan2 in reverse() limits the achievable precision
    assert metrics["yaw_mae_rad"] == pytest.approx(wrapped_yaw_err / 2, rel=1e-5)
    assert metrics["terminal_yaw_err_rad"] == pytest.approx(0.0)
    assert metrics["lin_vel_err_mps"] == pytest.approx(1.0)
    assert metrics["ang_vel_err_radps"] == pytest.approx(0.1)
    assert metrics["terminal_lin_vel_err_mps"] == pytest.approx(1.0)
    assert metrics["terminal_ang_vel_err_radps"] == pytest.approx(0.1)
    assert metrics["unit_circle_err"] == pytest.approx((0.36 + 0.75) / 2)
    assert "kin_pos_err_m" not in metrics


def _consistent_trajectory(twist, lin_velocities, omegas, delta_t=DELTA_T):
    """Trapezoid-consistent (1, K, 7) trajectory from an origin pose and
    the given twist, matching the kinematic model of the hook."""
    velocities = [tuple(twist[:2])] + list(lin_velocities)
    omegas = [twist[2]] + list(omegas)
    xy, yaw = [(0.0, 0.0)], [0.0]
    for k in range(len(lin_velocities)):
        xy.append(
            tuple(
                xy[-1][axis]
                + delta_t / 2 * (velocities[k][axis] + velocities[k + 1][axis])
                for axis in range(2)
            )
        )
        yaw.append(yaw[-1] + delta_t / 2 * (omegas[k] + omegas[k + 1]))
    states = [
        [x, y, math.sin(a), math.cos(a), vx, vy, w]
        for (x, y), a, (vx, vy), w in zip(xy[1:], yaw[1:], velocities[1:], omegas[1:])
    ]
    return torch.tensor(states, dtype=torch.float32).unsqueeze(0)


def test_kinematic_errors_zero_for_consistent_trajectory():
    hook = AquaTrajectoryEvalHook(delta_t=DELTA_T)
    twist = torch.tensor([0.2, 0.1, 0.3])
    pred7 = _consistent_trajectory(
        twist, [(0.4, 0.2), (0.6, 0.1), (0.5, 0.3)], [0.5, 0.7, 0.4]
    )
    gt7 = torch.zeros_like(pred7)

    metrics = hook._frame_metrics(pred7, gt7, twist)
    assert metrics["kin_pos_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["kin_yaw_err_rad"] == pytest.approx(0.0, abs=1e-6)

    pred7[0, 1, 0] += 0.05  # breaks step 1 and step 2 increments by +/-0.05
    metrics = hook._frame_metrics(pred7, gt7, twist)
    assert metrics["kin_pos_err_m"] == pytest.approx(2 * 0.05 / 3)
    assert metrics["kin_yaw_err_rad"] == pytest.approx(0.0, abs=1e-6)


def test_after_test_iter_accumulates_cpu_scalars_and_reports():
    hook = AquaTrajectoryEvalHook()
    model = ResetCountingModel()
    logs = []
    runner = types.SimpleNamespace(
        model=model, logger=types.SimpleNamespace(info=logs.append)
    )

    pred = torch.zeros(1, 2, 7)
    pred[..., 3] = 1.0
    pred_with_grad = pred.clone().requires_grad_(True)

    hook.before_test(runner)
    hook.after_test_iter(runner, 0, _batch("A"), pred, mode="test")
    hook.after_test_iter(runner, 1, _batch("A"), pred_with_grad, mode="test")

    assert hook._frames == 2
    assert all(isinstance(value, float) for value in hook._sums.values())
    assert hook._sums["ADE_m"] == pytest.approx(0.0)

    hook.after_test_epoch(runner)
    assert len(logs) == 1
    assert "Aqua trajectory evaluation over 2 frames" in logs[0]


def test_hook_registered_and_configured():
    import robonav  # noqa: F401
    from prefusion.registry import HOOKS as PREFUSION_HOOKS
    from robonav.registry import HOOKS

    assert HOOKS.get("AquaTrajectoryEvalHook") is AquaTrajectoryEvalHook
    assert (
        PREFUSION_HOOKS.get("robonav.AquaTrajectoryEvalHook") is AquaTrajectoryEvalHook
    )

    cfg = Config.fromfile(
        os.path.join(os.path.dirname(__file__), "..", "configs", "kinogoal_dla_resnet18_overfit.py")
    )
    assert cfg.custom_hooks[0]["type"] == "robonav.AquaTrajectoryEvalHook"
    assert cfg.test_dataset["batch_size"] == 1
