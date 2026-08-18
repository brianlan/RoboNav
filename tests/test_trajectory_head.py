import io

import pytest
import torch
import torch.nn as nn

from robonav.aqua.model.trajectory_head import TrajectoryHead
from robonav.registry import MODELS


def _head(**overrides):
    cfg = dict(
        feat_chans=8,
        hidden_chans=6,
        intermediate_chans=4,
        out_chans=7,
        num_trajectory_steps=5,
    )
    cfg.update(overrides)
    return TrajectoryHead(**cfg)


def _inputs(batch_size):
    return torch.randn(batch_size, 8), torch.randn(batch_size, 6)


def test_output_shape_for_representative_batch_sizes():
    for batch_size in (1, 3, 16):
        out = _head()(*_inputs(batch_size))
        assert out.shape == (batch_size, 5, 7)
        assert out.dtype == torch.float32


def test_signed_output_without_final_relu():
    head = _head()
    with torch.no_grad():
        head.out_conv.weight.zero_()
        head.out_conv.bias.fill_(-1.0)
    assert (head(*_inputs(3)) < 0).all()


def test_backward_gradient_connectivity():
    head = _head()
    final_feat, hidden = _inputs(4)
    final_feat.requires_grad_(True)
    hidden.requires_grad_(True)
    head(final_feat, hidden).sum().backward()

    for tensor in (final_feat, hidden):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    def _blocked(param):
        return param is head.temporal_conv1.weight or param is head.temporal_conv1.bias

    for name, param in head.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        if _blocked(param):
            # zero-initialized temporal_conv2 blocks gradient flow through the
            # residual branch, so temporal_conv1 receives exact zero gradients
            assert torch.count_nonzero(param.grad) == 0, name
        else:
            assert torch.count_nonzero(param.grad) > 0, name

    # with a nonzero residual branch every parameter receives gradient
    with torch.no_grad():
        head.temporal_conv2.weight.normal_()
    head.zero_grad()
    head(final_feat, hidden).sum().backward()
    for name, param in head.named_parameters():
        assert torch.count_nonzero(param.grad) > 0, name


def test_initialization_exact_calls(monkeypatch):
    records = []
    for name in ("kaiming_normal_", "xavier_uniform_", "zeros_"):
        original = getattr(torch.nn.init, name)

        def spy(tensor, _original=original, _name=name, **kwargs):
            records.append((_name, tensor, kwargs))
            _original(tensor, **kwargs)

        monkeypatch.setattr(torch.nn.init, name, spy)
    head = _head()

    kaiming = {id(t): kw for n, t, kw in records if n == "kaiming_normal_"}
    xavier = {id(t): kw for n, t, kw in records if n == "xavier_uniform_"}
    zeros = {id(t) for n, t, _ in records if n == "zeros_"}

    assert set(map(id, (head.compress.weight, head.temporal_conv1.weight))) == set(kaiming)
    assert kaiming[id(head.compress.weight)] == {"nonlinearity": "relu"}
    assert kaiming[id(head.temporal_conv1.weight)] == {"mode": "fan_out", "nonlinearity": "relu"}
    assert set(map(id, (head.to_seq.weight, head.out_conv.weight))) == set(xavier)
    assert all(kw == {} for kw in xavier.values())
    assert zeros == set(
        map(
            id,
            (
                head.compress.bias,
                head.to_seq.bias,
                head.temporal_conv1.bias,
                head.temporal_conv2.weight,
                head.temporal_conv2.bias,
                head.out_conv.bias,
            ),
        )
    )


def test_residual_branch_starts_as_identity():
    head = _head()
    final_feat, hidden = _inputs(3)
    with torch.no_grad():
        compressed = torch.relu(head.compress(torch.concat((final_feat, hidden), dim=1)))
        z0 = head.to_seq(compressed).reshape(3, head.intermediate_chans, 1, head.num_trajectory_steps)
        expected = head.out_conv(z0).permute(0, 3, 2, 1).squeeze(2)
    torch.testing.assert_close(head(final_feat, hidden), expected)


def test_conv2d_temporal_structure_and_no_conv1d_or_norm():
    head = _head()
    for layer in (head.temporal_conv1, head.temporal_conv2):
        assert isinstance(layer, nn.Conv2d)
        assert layer.kernel_size == (1, 3)
        assert layer.padding == (0, 1)
    assert isinstance(head.out_conv, nn.Conv2d)
    assert head.out_conv.kernel_size == (1, 1)
    assert head.out_conv.padding == (0, 0)
    assert not any(isinstance(m, nn.Conv1d) for m in head.modules())
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm))
        for m in head.modules()
    )


def test_onnx_export_transpose_consumes_rank4():
    onnx = pytest.importorskip("onnx")
    head = _head()
    head.eval()
    buf = io.BytesIO()
    torch.onnx.export(head, _inputs(2), buf, input_names=["final_feat", "hidden"])
    graph = onnx.load_from_string(buf.getvalue()).graph
    transposes = [n for n in graph.node if n.op_type == "Transpose"]
    assert transposes
    for node in transposes:
        perm = next(a.ints for a in node.attribute if a.name == "perm")
        assert len(perm) == 4


def test_registry_build():
    head = MODELS.build(
        dict(
            type="robonav.TrajectoryHead",
            feat_chans=8,
            hidden_chans=6,
            intermediate_chans=4,
            out_chans=7,
            num_trajectory_steps=5,
        )
    )
    assert isinstance(head, TrajectoryHead)
    assert head(*_inputs(2)).shape == (2, 5, 7)


if __name__ == "__main__":
    test_output_shape_for_representative_batch_sizes()
    test_signed_output_without_final_relu()
    test_backward_gradient_connectivity()
    test_residual_branch_starts_as_identity()
    test_conv2d_temporal_structure_and_no_conv1d_or_norm()
    test_onnx_export_transpose_consumes_rank4()
    test_registry_build()
