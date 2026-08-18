import torch

from robonav.aqua.model.temporal_fuser import TemporalFiLM, TemporalFuser

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
    x = torch.cat((f.vfeat_compressor(inputs[0]), *inputs[1:]), dim=1)
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
