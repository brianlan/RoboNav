import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["MultiScaleDepthLoss"]


@MODELS.register_module()
class MultiScaleDepthLoss(BaseModel):
    def __init__(self, beta=1.0, loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, predictions, target, valid_mask):
        if target.shape != valid_mask.shape:
            raise ValueError(
                "target and valid_mask must have exactly the same shape, "
                f"got {target.shape} and {valid_mask.shape}"
            )
        for index, prediction in enumerate(predictions):
            if prediction.shape[:2] != target.shape[:2]:
                raise ValueError(
                    f"prediction {index} batch/channel shape {prediction.shape[:2]} "
                    f"must match target {target.shape[:2]}"
                )
        valid = valid_mask.to(target.dtype)
        target = target.masked_fill(~valid_mask, 0)
        losses = []
        for prediction in predictions:
            size = prediction.shape[-2:]
            valid_ratio = F.interpolate(valid, size=size, mode="area")
            target_sum = F.interpolate(target, size=size, mode="area")
            scale_target = target_sum / valid_ratio.clamp_min(1e-12)
            loss = F.smooth_l1_loss(
                prediction, scale_target, beta=self.beta, reduction="none"
            )
            losses.append(
                (loss * valid_ratio).sum() / valid_ratio.sum().clamp_min(1e-12)
            )
        return sum(losses) / len(losses) * self.loss_weight
