import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["MultiScaleDepthLoss"]


@MODELS.register_module()
class MultiScaleDepthLoss(BaseModel):
    """Multi-scale valid-mask-weighted Smooth L1 in log-meter range.

    Predictions and targets are normalized depth in [0, 1]; both are mapped
    to meters via ``max_depth`` and compared as ``log(d + log_offset)``
    (docs/aquanet_loss_supervision_plan.md §6). Hard clipping semantics are
    supplied by the target/valid mask and are not modified here.
    """

    def __init__(self, max_depth, log_offset, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.log_offset = log_offset # preventing infinite grad when prediction approaching 0
        self.beta = beta

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
            log_prediction = (prediction * self.max_depth + self.log_offset).log()
            log_target = (scale_target * self.max_depth + self.log_offset).log()
            loss = F.smooth_l1_loss(
                log_prediction, log_target, beta=self.beta, reduction="none"
            )
            losses.append(
                (loss * valid_ratio).sum() / valid_ratio.sum().clamp_min(1e-12)
            )
        return sum(losses) / len(losses)
