import torch
import torch.nn as nn
from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["TemporalFuser"]


MODELS.register_module()
class TemporalFuser(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vfeat_compressor = SpatialFeatureCompressor()
        self.sru = SRU()
        self.temporal_film = TemporalFiLM()
        self.history_enhanced_compressor = SpatialFeatureCompressor()

    def forward(self, feat):
        scene_desc = self.vfeat_compressor(f3g)
        cur_state = torch.cat((scene_desc, cur_velo, delta_pose, goal), dim=1)
        hidden = self.sru(cur_state)
        f3g_history_injected = self.temporal_film(f3g, hidden)
        final_feat = self.history_enhanced_compressor(f3g_history_injected)


class SpatialFeatureCompressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat):
        return feat


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
