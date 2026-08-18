import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["FeatureModulation"]


@MODELS.register_module()
class FeatureModulation(BaseModel):
    def __init__(
        self,
        f4_chans,
        f3_chans,
        out_chans,
        *args,
        goal_chans=6,
        twist_chans=3,
        goal_scales=(10.0, 10.0, math.pi, 1.0, 1.0, 1.0),
        twist_scales=(1.0, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer("goal_scales", torch.tensor(goal_scales))
        self.register_buffer("twist_scales", torch.tensor(twist_scales))
        self.film_by_goal_n_twist = FiLMByGoalAndTwist(
            goal_chans + twist_chans, f4_chans
        )
        self.f4_conv_1x1 = nn.Conv2d(f4_chans, out_chans, 1)
        self.f3_conv_1x1 = nn.Conv2d(f3_chans, out_chans, 1)
        self.spatial_gate = SpatialGate(goal_chans + twist_chans, out_chans)

    def forward(self, f4, f3, twist, goal):
        goal = goal / self.goal_scales
        twist = twist / self.twist_scales
        f4m = self.film_by_goal_n_twist(f4, goal, twist)
        f4m_up = F.interpolate(
            self.f4_conv_1x1(f4m), size=f3.shape[-2:], mode="nearest"
        )
        f3_fused = F.relu(f4m_up + self.f3_conv_1x1(f3))
        return self.spatial_gate(f3_fused, goal, twist)


class FiLMByGoalAndTwist(nn.Module):
    def __init__(self, in_chans, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.gamma_linear = nn.Linear(in_chans, out_chans)
        self.beta_linear = nn.Linear(in_chans, out_chans)
        nn.init.zeros_(self.gamma_linear.weight)
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)

    def forward(self, feat, goal, twist):
        goal_n_twist = torch.cat((goal, twist), dim=1)
        delta_gamma = self.gamma_linear(goal_n_twist)
        beta = self.beta_linear(goal_n_twist)
        return feat + feat * delta_gamma[:, :, None, None] + beta[:, :, None, None]


class SpatialGate(nn.Module):
    def __init__(self, state_chans, feat_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_linear = nn.Linear(state_chans, feat_chans)
        self.adjust_conv_1x1 = nn.Conv2d(feat_chans, feat_chans, 1)
        self.weight_conv_1x1 = nn.Conv2d(feat_chans, feat_chans, 1)
        self.sigmoid = nn.Hardsigmoid()
        self.feat_delta_conv_1x1 = nn.Conv2d(feat_chans, feat_chans, 1)
        nn.init.zeros_(self.feat_delta_conv_1x1.weight)
        nn.init.zeros_(self.feat_delta_conv_1x1.bias)

    def forward(self, feat, goal, twist):
        goal_n_twist = torch.cat((goal, twist), dim=1)
        state = self.state_linear(goal_n_twist)
        adjusted = self.adjust_conv_1x1(feat)
        added = torch.relu(adjusted + state[:, :, None, None])
        weight = self.sigmoid(self.weight_conv_1x1(added))
        feat_delta = self.feat_delta_conv_1x1(feat)
        return feat + weight * feat_delta
