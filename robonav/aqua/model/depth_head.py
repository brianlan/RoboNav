import torch
import torch.nn as nn
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS


__all__ = ["DepthHead"]


@MODELS.register_module()
class DepthHead(BaseModel):
    def __init__(
        self, f4_chans, f3_chans, f2_chans, f1_chans, decoder_chans, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.laterals = nn.ModuleList(
            nn.Conv2d(chans, decoder_chans, 1)
            for chans in (f4_chans, f3_chans, f2_chans, f1_chans)
        )
        self.refinements = nn.ModuleList(
            nn.Conv2d(decoder_chans, decoder_chans, 3, padding=1)
            for _ in range(4)
        )
        self.predictors = nn.ModuleList(
            nn.Conv2d(decoder_chans, 1, 1) for _ in range(4)
        )
        self._init_weights()

    def _init_weights(self):
        for lateral in self.laterals:
            nn.init.xavier_uniform_(lateral.weight)
            nn.init.zeros_(lateral.bias)
        for refinement in self.refinements:
            nn.init.kaiming_normal_(
                refinement.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(refinement.bias)
        for predictor in self.predictors:
            nn.init.xavier_uniform_(predictor.weight)
            nn.init.zeros_(predictor.bias)

    def forward(self, f4, f3, f2, f1):
        features = (f4, f3, f2, f1)
        fused = self.laterals[0](f4)
        predictions = []
        for index, feature in enumerate(features):
            if index:
                fused = self.laterals[index](feature) + F.interpolate(
                    fused, size=feature.shape[-2:], mode="nearest"
                )
            refined = F.relu(self.refinements[index](fused))
            predictions.append(torch.sigmoid(self.predictors[index](refined)))
        return tuple(predictions)
