import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from prefusion.models import BaseModel

from robonav.registry import MODELS
from robonav.aqua.model.sru import SRU

__all__ = ["TemporalFuser"]


@MODELS.register_module()
class TemporalFuser(BaseModel):
    def __init__(
        self,
        feat_chans,
        hidden_chans,
        *args,
        twist_chans=3,
        delta_pose_chans=3,
        goal_chans=6,
        goal_scales=(10.0, 10.0, math.pi, 1.0, 1.0, 1.0),
        twist_scales=(1.0, 1.0, 1.0),
        delta_pose_scales=(0.1, 0.1, 0.1),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vfeat_compressor = SpatialFeatureCompressor(feat_chans)
        self.sru = SRU(
            feat_chans + twist_chans + delta_pose_chans + goal_chans, hidden_chans
        )
        self.temporal_film = TemporalFiLM(hidden_chans, feat_chans)
        self.history_enhanced_compressor = SpatialFeatureCompressor(feat_chans)
        self.register_buffer("goal_scales", torch.tensor(goal_scales))
        self.register_buffer("twist_scales", torch.tensor(twist_scales))
        self.register_buffer("delta_pose_scales", torch.tensor(delta_pose_scales))
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, f3g, twist, delta_pose, goal):
        scene_desc = self.vfeat_compressor(f3g)
        cur_state = torch.cat(
            (
                scene_desc,
                twist / self.twist_scales,
                delta_pose / self.delta_pose_scales,
                goal / self.goal_scales,
            ),
            dim=1,
        )
        _, self.state = self.sru(cur_state.unsqueeze(0), self.state)
        hidden = self.state[0][-1]
        f3g_history_injected = self.temporal_film(f3g, hidden)
        final_feat = self.history_enhanced_compressor(f3g_history_injected)
        return final_feat, hidden


class SpatialFeatureCompressor(nn.Module):
    def __init__(self, num_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(num_channels, num_channels, (6, 8))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, feat):
        feat = F.relu(self.conv1(feat))
        feat = F.relu(self.conv2(feat))
        feat = self.conv3(feat)
        return feat.squeeze(-1).squeeze(-1)


class TemporalFiLM(nn.Module):
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

    def forward(self, feat, hidden):
        delta_gamma = self.gamma_linear(hidden)
        beta = self.beta_linear(hidden)
        return feat + feat * delta_gamma[:, :, None, None] + beta[:, :, None, None]
