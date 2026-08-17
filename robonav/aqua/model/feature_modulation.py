import torch
import torch.nn as nn
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["FeatureModulation"]


@MODELS.register_module()
class FeatureModulation(BaseModel):
    def __init__(self, f4_chans, f3_chans, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film_by_goal_n_state = FiLMByGoalAndState(f4_chans, f4_chans)
        self.spatial_gate = SpatialGate()
        self.f4_conv_1x1 = nn.Conv2d(f4_chans, out_chans, 1)
        self.f3_conv_1x1 = nn.Conv2d(f3_chans, out_chans, 1)

    def forward(self, f4, f3, velo, goal):
        f4m = self.film_by_goal_n_state(f4, goal, velo)
        f4m_up = F.interpolate(self.f4_conv_1x1(f4m), scale_factor=2, mode="nearest")
        f3_fused = F.relu(f4m_up + self.f3_conv_1x1(f3))
        f3g = self.spatial_gate(f3_fused, goal, velo)
        return f3g


class FiLMByGoalAndState(nn.Module):
    def __init__(self, in_chans, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.gamma_linear = nn.Linear(in_chans, out_chans)
        self.beta_linear = nn.Linear(in_chans, out_chans)

    def forward(self, feat, goal, state):
        goal_n_state = torch.cat((goal, state), dim=1)
        delta_gamma = self.gamma_linear(goal_n_state)
        beta = self.beta_linear(goal_n_state)
        return feat + feat * delta_gamma + beta


class SpatialGate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat, goal, state):
        return feat

