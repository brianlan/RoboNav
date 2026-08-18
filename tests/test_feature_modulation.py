import math

import torch

from robonav.aqua.model.feature_modulation import (
    FeatureModulation,
    FiLMByGoalAndTwist,
    SpatialGate,
)

B, F4_CHANS, F3_CHANS, OUT_CHANS = 2, 5, 7, 6
H4, W4, H3, W3 = 5, 7, 9, 12  # f3 spatial size is not exactly 2x f4

DEFAULT_GOAL_SCALES = (10.0, 10.0, math.pi, 1.0, 1.0, 1.0)
DEFAULT_TWIST_SCALES = (1.0, 1.0, 1.0)


def make_inputs():
    return dict(
        f4=torch.randn(B, F4_CHANS, H4, W4),
        f3=torch.randn(B, F3_CHANS, H3, W3),
        twist=torch.randn(B, 3),
        goal=torch.randn(B, 6),
    )


def test_output_shape_with_distinct_chans_and_non_2x_f3():
    torch.manual_seed(0)
    m = FeatureModulation(f4_chans=F4_CHANS, f3_chans=F3_CHANS, out_chans=OUT_CHANS)
    out = m(**make_inputs())
    assert out.shape == (B, OUT_CHANS, H3, W3)


def test_backward_finite_grads_for_all_params():
    torch.manual_seed(1)
    m = FeatureModulation(f4_chans=F4_CHANS, f3_chans=F3_CHANS, out_chans=OUT_CHANS)
    m(**make_inputs()).sum().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_scale_buffers_registered_with_defaults():
    m = FeatureModulation(f4_chans=F4_CHANS, f3_chans=F3_CHANS, out_chans=OUT_CHANS)
    buffers = dict(m.named_buffers())
    assert torch.equal(buffers["goal_scales"], torch.tensor(DEFAULT_GOAL_SCALES))
    assert torch.equal(buffers["twist_scales"], torch.tensor(DEFAULT_TWIST_SCALES))


def test_normalized_goal_and_twist_reach_both_branches():
    torch.manual_seed(2)
    goal_scales = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    twist_scales = (3.0, 9.0, 27.0)
    m = FeatureModulation(
        f4_chans=F4_CHANS,
        f3_chans=F3_CHANS,
        out_chans=OUT_CHANS,
        goal_scales=goal_scales,
        twist_scales=twist_scales,
    )
    captured = {}

    def record(name):
        def hook(module, args):
            captured[name] = (args[1], args[2])

        return hook

    m.film_by_goal_n_twist.register_forward_pre_hook(record("film"))
    m.spatial_gate.register_forward_pre_hook(record("gate"))
    inputs = make_inputs()
    m(**inputs)
    expected_goal = inputs["goal"] / torch.tensor(goal_scales)
    expected_twist = inputs["twist"] / torch.tensor(twist_scales)
    for name in ("film", "gate"):
        torch.testing.assert_close(captured[name][0], expected_goal)
        torch.testing.assert_close(captured[name][1], expected_twist)


def test_fresh_film_is_identity():
    torch.manual_seed(3)
    film = FiLMByGoalAndTwist(9, F4_CHANS)
    feat = torch.randn(B, F4_CHANS, H4, W4)
    goal, twist = torch.randn(B, 6), torch.randn(B, 3)
    torch.testing.assert_close(film(feat, goal, twist), feat)


def test_fresh_spatial_gate_is_identity():
    torch.manual_seed(4)
    gate = SpatialGate(9, OUT_CHANS)
    feat = torch.randn(B, OUT_CHANS, H3, W3)
    goal, twist = torch.randn(B, 6), torch.randn(B, 3)
    torch.testing.assert_close(gate(feat, goal, twist), feat)
