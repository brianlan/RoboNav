import torch
import torch.nn as nn
from prefusion.models import BaseModel

from robonav.registry import MODELS
from robonav.aqua.model.sru import SRU

__all__ = ["TemporalFuser"]


MODELS.register_module()


class TemporalFuser(BaseModel):
    def __init__(
        self,
        feat_chans,
        hidden_chans,
        *args,
        twist_chans=3,
        delta_pose_chans=3,
        goal_chans=6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vfeat_compressor = SpatialFeatureCompressor(feat_chans)
        self.sru = SRU(
            feat_chans + twist_chans + delta_pose_chans + goal_chans, hidden_chans
        )
        self.temporal_film = TemporalFiLM(hidden_chans, feat_chans)
        self.history_enhanced_compressor = SpatialFeatureCompressor(feat_chans)
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, f3g, twist, delta_pose, goal):
        scene_desc = self.vfeat_compressor(f3g)
        cur_state = torch.cat((scene_desc, twist, delta_pose, goal), dim=1)
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

    def forward(self, feat):
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        return feat.squeeze(-1).squeeze(-1)


class TemporalFiLM(nn.Module):
    def __init__(self, in_chans, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.gamma_linear = nn.Linear(in_chans, out_chans)
        self.beta_linear = nn.Linear(in_chans, out_chans)

    def forward(self, feat, hidden):
        delta_gamma = self.gamma_linear(hidden)
        beta = self.beta_linear(hidden)
        return feat + feat * delta_gamma[:, :, None, None] + beta[:, :, None, None]
