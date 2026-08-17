import torch
import torch.nn as nn
from prefusion.models import BaseModel

from robonav.registry import MODELS
from robonav.aqua.model.sru import SRU

__all__ = ["TemporalFuser"]


MODELS.register_module()
class TemporalFuser(BaseModel):
    def __init__(self, input_chans, hidden_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vfeat_compressor = SpatialFeatureCompressor(input_chans)
        self.sru = SRU(input_chans, hidden_chans)
        self.temporal_film = TemporalFiLM(hidden_chans, hidden_chans)
        self.history_enhanced_compressor = SpatialFeatureCompressor(input_chans)
        self.hidden = self.cell = None

    def forward(self, f3g, velo, delta_pose, goal):
        scene_desc = self.vfeat_compressor(f3g)
        cur_state = torch.cat((scene_desc, velo, delta_pose, goal), dim=1)
        self.hidden, self.cell = self.sru(cur_state, (self.hidden, self.cell))
        f3g_history_injected = self.temporal_film(f3g, self.hidden)
        final_feat = self.history_enhanced_compressor(f3g_history_injected)
        return final_feat, self.hidden


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
        goal_n_state = torch.cat(hidden, dim=1)
        delta_gamma = self.gamma_linear(goal_n_state)
        beta = self.beta_linear(goal_n_state)
        return feat + feat * delta_gamma + beta
