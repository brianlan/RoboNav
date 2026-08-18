import os

import torch
import torch.nn as nn
from mmengine import Config

from robonav.aqua.loss import MultiScaleDepthLoss
from robonav.aqua.model.aqua import AquaNet
from robonav.aqua.model.depth_head import DepthHead
from robonav.registry import MODELS


def _head():
    return DepthHead(
        f4_chans=16,
        f3_chans=12,
        f2_chans=8,
        f1_chans=4,
        decoder_chans=6,
    )


def _features(requires_grad=False):
    return tuple(
        torch.randn(shape, requires_grad=requires_grad)
        for shape in (
            (2, 16, 3, 4),
            (2, 12, 5, 7),
            (2, 8, 9, 13),
            (2, 4, 17, 25),
        )
    )


def test_odd_pyramid_outputs_and_input_gradients():
    torch.manual_seed(0)
    features = _features(requires_grad=True)
    head = _head()
    predictions = head(*features)

    assert [prediction.shape for prediction in predictions] == [
        (2, 1, 3, 4),
        (2, 1, 5, 7),
        (2, 1, 9, 13),
        (2, 1, 17, 25),
    ]
    assert all(torch.all((prediction >= 0) & (prediction <= 1)) for prediction in predictions)

    target = torch.full((2, 1, 17, 25), 0.25)
    valid = torch.ones_like(target, dtype=torch.bool)
    MultiScaleDepthLoss(beta=0.1)(predictions, target, valid).backward()
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad) > 0
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_layer_contract_and_exact_initialization(monkeypatch):
    records = []
    for name in ("kaiming_normal_", "xavier_uniform_", "zeros_"):
        original = getattr(torch.nn.init, name)

        def spy(tensor, _original=original, _name=name, **kwargs):
            records.append((_name, tensor, kwargs))
            return _original(tensor, **kwargs)

        monkeypatch.setattr(torch.nn.init, name, spy)

    head = _head()
    convs = [module for module in head.modules() if isinstance(module, nn.Conv2d)]
    assert convs == [
        *head.laterals,
        *head.refinements,
        *head.predictors,
    ]
    assert [conv.kernel_size for conv in head.laterals] == [(1, 1)] * 4
    assert [conv.kernel_size for conv in head.refinements] == [(3, 3)] * 4
    assert [conv.kernel_size for conv in head.predictors] == [(1, 1)] * 4
    assert not any(
        isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
        for module in head.modules()
    )

    kaiming = {id(tensor): kwargs for name, tensor, kwargs in records if name == "kaiming_normal_"}
    xavier = {id(tensor): kwargs for name, tensor, kwargs in records if name == "xavier_uniform_"}
    zeros = {id(tensor) for name, tensor, _ in records if name == "zeros_"}
    assert kaiming == {
        id(layer.weight): {"mode": "fan_out", "nonlinearity": "relu"}
        for layer in head.refinements
    }
    assert xavier == {
        id(layer.weight): {}
        for layer in (*head.laterals, *head.predictors)
    }
    assert zeros == {
        id(layer.bias)
        for layer in (*head.laterals, *head.refinements, *head.predictors)
    }


def test_registry_and_overfit_config_build():
    config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__), "..", "configs", "kinogoal_dla_resnet18_overfit.py"
        )
    )
    head = MODELS.build(config.model.depth_head)
    loss = MODELS.build(config.model.depth_loss)
    assert isinstance(head, DepthHead)
    assert head.laterals[-1].in_channels == 64
    assert head.laterals[-1].out_channels == 64
    assert loss.beta == 0.1
    assert config.possible_sequence_lengths == [10]
    assert config.transformables.camera_depths.tensor_smith.max_depth == 5


class _Backbone(nn.Module):
    def forward(self, images, pe):
        base = images.mean(dim=1, keepdim=True)
        f1 = nn.functional.avg_pool2d(base, 2).repeat(1, 4, 1, 1)
        f2 = nn.functional.avg_pool2d(f1, 2)[:, :4]
        f3 = nn.functional.avg_pool2d(f2, 2)
        f4 = nn.functional.avg_pool2d(f3, 2)
        return f1, f2, f3, f4


class _FeatureModulation(nn.Module):
    def forward(self, f4, f3, twist, goal):
        return f3


class _TemporalFuser(nn.Module):
    def forward(self, feature, twist, delta_poses, goal):
        pooled = feature.mean((2, 3))
        return pooled, pooled


class _TrajectoryHead(nn.Module):
    def forward(self, final_feat, hidden):
        return final_feat


def test_aquanet_loss_backward_reaches_depth_head(monkeypatch):
    build = MODELS.build
    monkeypatch.setattr(MODELS, "build", lambda cfg: None if cfg is None else build(cfg))
    model = AquaNet(
        depth_head=dict(
            type="robonav.DepthHead",
            f4_chans=4,
            f3_chans=4,
            f2_chans=4,
            f1_chans=4,
            decoder_chans=4,
        ),
        depth_loss=dict(type="robonav.MultiScaleDepthLoss", beta=0.1),
    )
    model.backbone = _Backbone()
    model.feature_modulation = _FeatureModulation()
    model.temporal_fuser = _TemporalFuser()
    model.trajectory_head = _TrajectoryHead()

    loss_dict = model(
        camera_images=[torch.randn(1, 3, 32, 32)],
        camera_depths=[torch.full((1, 1, 32, 32), 0.7)],
        camera_depth_valid_masks=[torch.ones(1, 1, 32, 32, dtype=torch.bool)],
        position_embedding=[torch.zeros(1, 6, 16, 16)],
        goal=[torch.zeros(1, 6)],
        twist=[torch.zeros(1, 3)],
        delta_poses=[torch.zeros(1, 3)],
        mode="loss",
    )
    assert set(loss_dict) == {"loss_depth"}
    loss_dict["loss_depth"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.depth_head.parameters()
    )
