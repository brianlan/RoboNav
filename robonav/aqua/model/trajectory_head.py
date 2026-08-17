import torch
import torch.nn as nn
from prefusion.models import BaseModel

from robonav.registry import MODELS


__all__ = ["TrajectoryHead"]


@MODELS.register_module()
class TrajectoryHead(BaseModel):
    def __init__(
        self,
        feat_chans,
        hidden_chans,
        intermediate_chans,
        out_chans,
        num_trajectory_steps,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intermediate_chans = intermediate_chans
        self.num_trajectory_steps = num_trajectory_steps
        self.compress = nn.Linear(feat_chans + hidden_chans, intermediate_chans)
        self.to_seq = nn.Linear(intermediate_chans, num_trajectory_steps * out_chans)
        self.temporal_conv = nn.Conv1d(intermediate_chans, intermediate_chans, 3, padding=1)
        self.out_conv = nn.Conv1d(intermediate_chans, out_chans, 3, padding=1)

    def forward(self, final_feat, hidden):
        compressed = torch.relu(self.compress(torch.concat((final_feat, hidden))))
        seq_feat = self.to_seq(compressed)
        z0 = seq_feat.reshape(seq_feat.shape[0], self.intermediate_chans, self.num_trajectory_steps)
        z1 = torch.relu(self.temporal_conv(z0))
        z2 = self.out_conv(z1)
        trajectory = z0 + z2
        return trajectory
