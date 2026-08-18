import math

import torch

from robonav.aqua.model.temporal_fuser import (
    SpatialFeatureCompressor,
    TemporalFiLM,
    TemporalFuser,
)

B, INPUT_CHANS, HIDDEN_CHANS, H, W = 2, 5, 7, 24, 32


def make_inputs():
    return (
        torch.randn(B, INPUT_CHANS, H, W),
        torch.randn(B, 3),
        torch.randn(B, 3),
        torch.randn(B, 6),
    )


def test_shapes_with_input_ne_hidden():
    torch.manual_seed(0)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    final_feat, hidden = f(*make_inputs())
    assert final_feat.shape == (B, INPUT_CHANS)
    assert hidden.shape == (B, HIDDEN_CHANS)
    assert f.state[0].shape == (1, B, HIDDEN_CHANS)
    assert f.state[1].shape == (1, B, HIDDEN_CHANS)


def test_two_steps_match_sru_unroll():
    torch.manual_seed(1)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    inputs = make_inputs()
    feat1, hidden1 = f(*inputs)
    feat2, hidden2 = f(*inputs)
    assert not torch.allclose(hidden2, hidden1)

    # Oracle: run the fuser's own SRU over the same 2-step sequence.
    x = torch.cat(
        (
            f.vfeat_compressor(inputs[0]),
            inputs[1] / f.twist_scales,
            inputs[2] / f.delta_pose_scales,
            inputs[3] / f.goal_scales,
        ),
        dim=1,
    )
    out_seq, (h, _) = f.sru(torch.stack([x, x]))
    torch.testing.assert_close(hidden1, out_seq[0])
    torch.testing.assert_close(hidden2, h[-1])
    expected_feat2 = f.history_enhanced_compressor(f.temporal_film(inputs[0], h[-1]))
    torch.testing.assert_close(feat2, expected_feat2)


def test_reset_reproducibility():
    torch.manual_seed(2)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    inputs = make_inputs()
    first = f(*inputs)
    f(*inputs)
    f.reset()
    assert f.state is None
    torch.testing.assert_close(f(*inputs), first)


def test_film_broadcast():
    torch.manual_seed(3)
    film = TemporalFiLM(HIDDEN_CHANS, INPUT_CHANS)
    feat = torch.randn(B, INPUT_CHANS, 6, 8)
    hidden = torch.randn(B, HIDDEN_CHANS)
    out = film(feat, hidden)
    gamma = film.gamma_linear(hidden).view(B, INPUT_CHANS, 1, 1)
    beta = film.beta_linear(hidden).view(B, INPUT_CHANS, 1, 1)
    torch.testing.assert_close(out, feat + feat * gamma + beta)


def test_gradients_flow_across_steps():
    torch.manual_seed(4)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    feat1, _ = f(*make_inputs())
    feat2, hidden2 = f(*make_inputs())
    (feat1.sum() + feat2.sum() + hidden2.sum()).backward()
    for p in f.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_compressor_relu_placement_and_contract():
    torch.manual_seed(5)
    c = SpatialFeatureCompressor(INPUT_CHANS)
    feat = torch.randn(B, INPUT_CHANS, H, W)
    assert c(feat).shape == (B, INPUT_CHANS)
    x = torch.relu(c.conv1(feat))
    x = torch.relu(c.conv2(x))
    torch.testing.assert_close(c(feat), c.conv3(x).squeeze(-1).squeeze(-1))


def test_compressor_no_activation_after_conv3():
    c = SpatialFeatureCompressor(INPUT_CHANS)
    with torch.no_grad():
        c.conv3.weight.zero_()
        c.conv3.bias.fill_(-1.0)
    out = c(torch.randn(B, INPUT_CHANS, H, W))
    assert (out < 0).all()


def test_compressor_kaiming_init(monkeypatch):
    calls = []

    def spy_kaiming(tensor, **kwargs):
        calls.append((tensor, kwargs))

    monkeypatch.setattr(torch.nn.init, "kaiming_normal_", spy_kaiming)
    torch.manual_seed(6)
    c = SpatialFeatureCompressor(INPUT_CHANS)
    assert len(calls) == 2
    assert calls[0][0] is c.conv1.weight
    assert calls[1][0] is c.conv2.weight
    assert all(
        kwargs["nonlinearity"] == "relu" and kwargs["mode"] == "fan_out"
        for _, kwargs in calls
    )
    assert c.conv1.bias.eq(0).all() and c.conv2.bias.eq(0).all()


def test_film_starts_as_exact_identity():
    torch.manual_seed(7)
    film = TemporalFiLM(HIDDEN_CHANS, INPUT_CHANS)
    for p in (
        film.gamma_linear.weight,
        film.gamma_linear.bias,
        film.beta_linear.weight,
        film.beta_linear.bias,
    ):
        assert p.eq(0).all()
    feat = torch.randn(B, INPUT_CHANS, H, W)
    hidden = torch.randn(B, HIDDEN_CHANS)
    torch.testing.assert_close(film(feat, hidden), feat)


def test_scale_buffer_defaults():
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    buffers = dict(f.named_buffers())
    torch.testing.assert_close(
        buffers["goal_scales"], torch.tensor([10.0, 10.0, math.pi, 1.0, 1.0, 1.0])
    )
    torch.testing.assert_close(buffers["twist_scales"], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(
        buffers["delta_pose_scales"], torch.tensor([0.1, 0.1, 0.1])
    )


def test_normalized_inputs_reach_sru_in_concat_order():
    torch.manual_seed(8)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    recorded = {}
    f.sru.register_forward_pre_hook(lambda module, args: recorded.update(x=args[0]))
    inputs = make_inputs()
    f(*inputs)
    expected = torch.cat(
        (
            f.vfeat_compressor(inputs[0]),
            inputs[1] / f.twist_scales,
            inputs[2] / f.delta_pose_scales,
            inputs[3] / f.goal_scales,
        ),
        dim=1,
    ).unsqueeze(0)
    torch.testing.assert_close(recorded["x"], expected)


def test_multiframe_backward_finite_grads():
    torch.manual_seed(9)
    f = TemporalFuser(feat_chans=INPUT_CHANS, hidden_chans=HIDDEN_CHANS)
    f.reset()
    losses = []
    for _ in range(4):
        feat, hidden = f(*make_inputs())
        losses.append(feat.sum() + hidden.sum())
    torch.stack(losses).mean().backward()
    for p in f.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
