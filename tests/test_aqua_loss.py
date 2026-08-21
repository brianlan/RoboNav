import math
import os

import pytest
import torch
import torch.nn.functional as F
from mmengine import Config

from robonav.aqua.loss import AquaLoss, MultiScaleDepthLoss
from robonav.registry import MODELS

DELTA_T = 0.1
TERMS = ("traj_xy", "traj_yaw", "traj_unit", "traj_vel", "kin_pos", "kin_yaw", "depth")
DEFAULT_WEIGHTS = {name: 1.0 for name in TERMS}
DEFAULT_DEPTH_LOSS = dict(
    type="robonav.MultiScaleDepthLoss", max_depth=5, log_offset=0.1, beta=1.0
)


def _loss(weights=None, depth_loss=None, **overrides):
    return AquaLoss(
        loss_weights=weights or DEFAULT_WEIGHTS,
        depth_loss=depth_loss or DEFAULT_DEPTH_LOSS,
        delta_t=DELTA_T,
        **overrides,
    )


def _integrate(twist, lin_velocities, omegas, delta_t=DELTA_T):
    """Exactly trapezoid-consistent [1, K, 7] trajectory in the current
    body frame: positions and yaws follow the same trapezoidal rule the
    kinematic losses assume, starting from p0=[0, 0], q0=[0, 1]."""
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
        [x, y, math.sin(angle), math.cos(angle), vx, vy, omega]
        for (x, y), angle, (vx, vy), omega in zip(
            xy[1:], yaw[1:], velocities[1:], omegas[1:]
        )
    ]
    return torch.tensor(states, dtype=torch.float32).unsqueeze(0)


def _depth_case():
    target = torch.tensor([[[[0.5, 1.0], [0.0, 1.0]]]])
    valid_mask = torch.tensor([[[[True, True], [False, True]]]])
    predictions = (
        torch.full((1, 1, 1, 1), 0.9, requires_grad=True),
        torch.tensor([[[[0.25, 0.5], [0.75, 1.0]]]], requires_grad=True),
    )
    return predictions, target, valid_mask


def test_registry_and_overfit_config_construction():
    config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__), "..", "robonav", "aqua", "configs", "kinogoal_dla_resnet18_overfit.py"
        )
    )
    loss = MODELS.build(config.model.loss)
    assert isinstance(loss, AquaLoss)
    assert loss.loss_weights == dict(config.loss_weights)
    assert set(config.loss_weights) == set(TERMS)
    assert isinstance(loss.depth_loss, MultiScaleDepthLoss)
    assert (
        loss.depth_loss.max_depth
        == config.transformables.camera_depths.tensor_smith.max_depth
    )
    assert loss.delta_t == 0.1
    assert loss.terminal_weight == config.model.loss.terminal_weight
    assert "depth_loss" not in config.model  # old weight path fully migrated
    with pytest.raises(ValueError, match="loss_weights keys must be"):
        AquaLoss(
            loss_weights=dict.fromkeys(TERMS[:-1]),
            depth_loss=DEFAULT_DEPTH_LOSS,
        )


def test_perfect_predictions_give_zero_losses_and_metrics():
    twist = (0.5, -0.2, 0.3)
    target = _integrate(
        twist,
        [(0.5 + 0.02 * k, -0.2) for k in range(20)],
        [0.3 + 0.01 * k for k in range(20)],
    )
    trajectory = target.clone().requires_grad_(True)
    depth_target = torch.tensor([[[[0.5, 1.0], [0.0, 1.0]]]])
    valid_mask = torch.tensor([[[[True, True], [False, True]]]])
    predictions = (
        torch.full((1, 1, 1, 1), 5.0 / 6.0, requires_grad=True),  # masked area mean
        torch.tensor([[[[0.5, 1.0], [0.0, 1.0]]]], requires_grad=True),
    )
    loss_dict = _loss(terminal_weight=5.0)(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=valid_mask,
    )
    for name in TERMS:
        assert loss_dict[f"loss_{name}"].abs().item() < 1e-5, name
    for name in (
        "velocity_error_moving",
        "velocity_error_stopped",
        "terminal_velocity_error",
        "ADE",
        "FDE",
        "yaw_error",
    ):
        assert loss_dict[name].abs().item() < 1e-5, name
    total, _ = _loss().parse_losses(loss_dict)
    assert total.abs().item() < 1e-4


def test_exact_weighted_composition_without_double_counting():
    weights = dict(
        traj_xy=2.0,
        traj_yaw=3.0,
        traj_unit=0.5,
        traj_vel=1.5,
        kin_pos=4.0,
        kin_yaw=0.25,
        depth=0.75,
    )
    twist = (0.2, 0.0, 0.1)
    target = _integrate(twist, [(0.4, 0.1)] * 20, [0.2] * 20)
    trajectory = (target + 0.05).detach().requires_grad_(True)
    predictions, depth_target, valid_mask = _depth_case()
    loss_dict = _loss(weights)(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=valid_mask,
    )

    assert not any("loss" in key for key in loss_dict if not key.startswith("loss_"))
    for name, weight in weights.items():
        torch.testing.assert_close(
            loss_dict[f"loss_{name}"], weight * loss_dict[f"{name}_raw"]
        )
        assert not loss_dict[f"{name}_raw"].requires_grad

    total, log_vars = _loss(weights).parse_losses(loss_dict)
    expected_total = sum(loss_dict[f"loss_{name}"] for name in TERMS)
    torch.testing.assert_close(total, expected_total)
    torch.testing.assert_close(log_vars["loss"], expected_total)


def test_yaw_wrap_continuity():
    twist = (0.0, 0.0, 1.6)
    # 20 steps of 1.6 rad/s cross the +-pi boundary; consistency holds on
    # sin/cos, which never wrap.
    target = _integrate(twist, [(0.3, 0.0)] * 20, [1.6] * 20)
    trajectory = target.clone().requires_grad_(True)
    predictions, depth_target, valid_mask = _depth_case()
    loss_dict = _loss()(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=valid_mask,
    )
    assert loss_dict["loss_kin_yaw"].abs().item() < 1e-5
    assert loss_dict["loss_traj_yaw"].abs().item() < 1e-5

    # metric yaw error must be wrap-safe: +3.13 and -3.13 are 0.0232 rad apart
    pred = target.clone()
    pred[..., 2:4] = torch.tensor([math.sin(3.13), math.cos(3.13)])
    wrap_target = target.clone()
    wrap_target[..., 2:4] = torch.tensor([math.sin(-3.13), math.cos(-3.13)])
    loss_dict = _loss()(
        trajectory=pred,
        trajectory_target=wrap_target,
        twist=torch.tensor([twist]),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=valid_mask,
    )
    assert loss_dict["yaw_error"].item() < 0.05
    assert loss_dict["traj_yaw_raw"].item() < 1e-3


def test_unit_circle_penalty_exact_value():
    twist = (0.0, 0.0, 0.0)
    target = _integrate(twist, [(0.0, 0.0)] * 4, [0.0] * 4)
    trajectory = target.clone()
    trajectory[..., 2:4] *= 2.0  # ||q|| = 2 -> (4 - 1)^2 = 9
    trajectory = trajectory.requires_grad_(True)
    loss_dict = _loss(weights=dict(DEFAULT_WEIGHTS, traj_unit=0.1))(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=_depth_case()[0],
        depth_target=_depth_case()[1],
        depth_valid_mask=_depth_case()[2],
    )
    torch.testing.assert_close(loss_dict["traj_unit_raw"], torch.tensor(9.0))
    torch.testing.assert_close(loss_dict["loss_traj_unit"], torch.tensor(0.9))
    assert loss_dict["traj_yaw_raw"].abs().item() < 1e-6  # direction is scale-free


def test_stopped_and_moving_metric_branches():
    twist = (0.0, 0.0, 0.0)
    target = _integrate(
        twist, [(0.0, 0.0)] * 3 + [(0.5, 0.0)] * 3, [0.0] * 3 + [0.1] * 3
    )
    trajectory = target.clone()
    trajectory[:, :3, 4] += 0.3  # vx error only on stopped steps
    trajectory[:, 3:, 6] += 0.6  # omega error only on moving steps
    # non-unit scales: metrics must be dimensionless like the loss, i.e.
    # stopped = (0.3 / 2) / 3, moving = (0.6 / 0.5) / 3, terminal = 0.6 / 0.5 / 3
    loss_dict = _loss(vel_scale=2.0, omega_scale=0.5)(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=_depth_case()[0],
        depth_target=_depth_case()[1],
        depth_valid_mask=_depth_case()[2],
    )
    torch.testing.assert_close(
        loss_dict["velocity_error_stopped"], torch.tensor(0.05)
    )
    torch.testing.assert_close(
        loss_dict["velocity_error_moving"], torch.tensor(0.4)
    )
    torch.testing.assert_close(
        loss_dict["terminal_velocity_error"], torch.tensor(0.4)
    )
    torch.testing.assert_close(loss_dict["stop_sample_ratio"], torch.tensor(0.5))


def test_current_frame_kinematics_with_nonzero_yaw():
    twist = (1.0, 0.2, 0.0)
    # yaw sweeps far past pi/2 while positions integrate v in the *current*
    # frame; rotating velocities by R(yaw) would create large residuals.
    target = _integrate(twist, [(1.0, 0.2)] * 20, [1.6] * 20)
    trajectory = target.clone().requires_grad_(True)
    loss_dict = _loss()(
        trajectory=trajectory,
        trajectory_target=target,
        twist=torch.tensor([twist]),
        depth_predictions=_depth_case()[0],
        depth_target=_depth_case()[1],
        depth_valid_mask=_depth_case()[2],
    )
    assert loss_dict["kin_pos_raw"].abs().item() < 1e-5
    assert loss_dict["kin_yaw_raw"].abs().item() < 1e-5


def test_interface_validates_semantic_shapes():
    twist = (0.0, 0.0, 0.0)
    target = _integrate(twist, [(0.1, 0.0)] * 4, [0.0] * 4)
    predictions, depth_target, valid_mask = _depth_case()
    mismatched = _integrate(twist, [(0.1, 0.0)] * 3, [0.0] * 3)
    with pytest.raises(ValueError, match="identical B,K,7 shapes"):
        _loss()(
            trajectory=target,
            trajectory_target=mismatched,
            twist=torch.zeros(1, 3),
            depth_predictions=predictions,
            depth_target=depth_target,
            depth_valid_mask=valid_mask,
        )
    with pytest.raises(ValueError, match="twist must have shape B,3"):
        _loss()(
            trajectory=target,
            trajectory_target=target,
            twist=torch.zeros(2, 3),
            depth_predictions=predictions,
            depth_target=depth_target,
            depth_valid_mask=valid_mask,
        )


def test_multiscale_log_depth_and_all_invalid_differentiable_zero():
    max_depth, log_offset, beta = 4.0, 0.5, 1.0
    depth_loss = dict(
        type="robonav.MultiScaleDepthLoss",
        max_depth=max_depth,
        log_offset=log_offset,
        beta=beta,
    )
    static = _integrate((0.0, 0.0, 0.0), [(0.0, 0.0)] * 4, [0.0] * 4)
    predictions, depth_target, valid_mask = _depth_case()

    def z(d):
        return (d * max_depth + log_offset).log()

    trajectory = _integrate((0.0, 0.0, 0.0), [(0.1, 0.0)] * 4, [0.0] * 4)
    trajectory = trajectory.clone().requires_grad_(True)
    loss_dict = _loss(depth_loss=depth_loss)(
        trajectory=trajectory,
        trajectory_target=static,
        twist=torch.zeros(1, 3),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=valid_mask,
    )
    # scale 1 (2x2): per-pixel masked target; only the 3 valid pixels count
    diffs_full = z(predictions[1]) - z(torch.tensor([[[[0.5, 1.0], [0.0, 1.0]]]]))
    per_pixel = F.smooth_l1_loss(diffs_full, torch.zeros_like(diffs_full), beta=beta, reduction="none")
    scale_full = (per_pixel * torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])).sum() / 3.0
    # scale 0 (1x1): masked area target = mean(0.5, 1.0, 1.0) / 3 valid = 5/6
    diffs_coarse = z(predictions[0]) - z(torch.tensor(5.0 / 6.0))
    scale_coarse = F.smooth_l1_loss(
        diffs_coarse, torch.zeros_like(diffs_coarse), beta=beta
    )
    torch.testing.assert_close(loss_dict["depth_raw"], (scale_full + scale_coarse) / 2)

    # all-invalid depth: finite differentiable zero; gradients still flow to trajectory
    trajectory = _integrate((0.0, 0.0, 0.0), [(0.1, 0.0)] * 4, [0.0] * 4)
    trajectory = trajectory.clone().requires_grad_(True)
    predictions = tuple(p.detach().clone().requires_grad_(True) for p in predictions)
    loss_dict = _loss(depth_loss=depth_loss)(
        trajectory=trajectory,
        trajectory_target=static,
        twist=torch.zeros(1, 3),
        depth_predictions=predictions,
        depth_target=depth_target,
        depth_valid_mask=torch.zeros_like(valid_mask),
    )
    assert loss_dict["loss_depth"] == 0
    assert torch.isfinite(loss_dict["loss_depth"])
    total, _ = _loss(depth_loss=depth_loss).parse_losses(loss_dict)
    total.backward()
    assert torch.isfinite(trajectory.grad).all()
    assert trajectory.grad.abs().sum() > 0
    for prediction in predictions:
        assert torch.isfinite(prediction.grad).all()
        assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))
