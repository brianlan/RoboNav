import math

import pytest
import torch
import torch.nn.functional as F

from robonav.aqua.loss import MultiScaleDepthLoss
from robonav.registry import MODELS


def test_target_and_mask_shapes_must_match():
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="target and valid_mask.*same shape"):
        MultiScaleDepthLoss(max_depth=5, log_offset=0.1)((torch.zeros(1, 1, 2, 2),), target, valid)


@pytest.mark.parametrize("prediction_shape", [(2, 1, 2, 2), (1, 2, 2, 2)])
def test_prediction_batch_and_channel_must_match_target(prediction_shape):
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    with pytest.raises(ValueError, match="prediction 0 batch/channel shape.*must match"):
        MultiScaleDepthLoss(max_depth=5, log_offset=0.1)((torch.zeros(prediction_shape),), target, valid)


def test_masked_area_target_has_valid_only_mean():
    predictions = tuple(torch.zeros(1, 1, 1, 1) for _ in range(4))
    target = torch.tensor([[[[0.0, 1.0], [0.8, 0.9]]]])
    valid = torch.tensor([[[[True, True], [False, False]]]])
    # max_depth=1, log_offset=1: masked area target = mean(0.0, 1.0) = 0.5,
    # so z_target = log(1.5) and z_prediction = log(1.0); beta=1 is in the
    # quadratic regime: 0.5 * log(1.5)^2.
    loss = MultiScaleDepthLoss(max_depth=1.0, log_offset=1.0, beta=1.0)(
        predictions, target, valid
    )
    torch.testing.assert_close(loss, torch.tensor(0.5 * math.log(1.5) ** 2))


def test_invalid_values_and_all_invalid_return_differentiable_zero():
    predictions = tuple(torch.full((1, 1, 1, 1), 0.4, requires_grad=True) for _ in range(4))
    target = torch.full((1, 1, 2, 2), float("inf"))
    valid = torch.zeros_like(target, dtype=torch.bool)
    loss = MultiScaleDepthLoss(max_depth=5, log_offset=0.1)(predictions, target, valid)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss == 0
    loss.backward()
    assert all(torch.equal(prediction.grad, torch.zeros_like(prediction)) for prediction in predictions)


def test_equal_scale_average_and_gradients():
    predictions = tuple(
        torch.full((1, 1, size, size), value, requires_grad=True)
        for size, value in zip((1, 2, 3, 4), (0.2, 0.4, 0.6, 0.8))
    )
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    loss = MultiScaleDepthLoss(max_depth=4.0, log_offset=1.0, beta=1.0)(
        predictions, target, valid
    )
    # every valid target is 0 m, so each scale compares log(4v + 1) vs log(1)
    diffs = [math.log(4 * value + 1) for value in (0.2, 0.4, 0.6, 0.8)]
    expected = sum(
        F.smooth_l1_loss(torch.tensor(d), torch.tensor(0.0), beta=1.0) for d in diffs
    ) / 4
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for prediction in predictions:
        assert torch.isfinite(prediction.grad).all()
        assert torch.count_nonzero(prediction.grad) > 0


def test_registry_build():
    loss = MODELS.build(
        dict(type="robonav.MultiScaleDepthLoss", max_depth=5, log_offset=0.1, beta=0.1)
    )
    assert isinstance(loss, MultiScaleDepthLoss)
    assert loss.max_depth == 5
    assert loss.log_offset == 0.1
    assert loss.beta == 0.1
