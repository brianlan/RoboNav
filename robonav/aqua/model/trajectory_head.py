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
        self.compress = nn.Linear(feat_chans + hidden_chans, feat_chans)
        self.to_seq = nn.Linear(feat_chans, num_trajectory_steps * intermediate_chans)
        self.temporal_conv1 = nn.Conv2d(intermediate_chans, intermediate_chans, (1, 3), padding=(0, 1))
        self.temporal_conv2 = nn.Conv2d(intermediate_chans, intermediate_chans, (1, 3), padding=(0, 1))
        self.out_conv = nn.Conv2d(intermediate_chans, out_chans, (1, 1))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.compress.weight, nonlinearity="relu")
        nn.init.zeros_(self.compress.bias)
        nn.init.xavier_uniform_(self.to_seq.weight)
        nn.init.zeros_(self.to_seq.bias)
        nn.init.kaiming_normal_(self.temporal_conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.temporal_conv1.bias)
        nn.init.zeros_(self.temporal_conv2.weight)
        nn.init.zeros_(self.temporal_conv2.bias)
        nn.init.xavier_uniform_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, final_feat, hidden):
        compressed = torch.relu(self.compress(torch.concat((final_feat, hidden), dim=1)))
        seq_feat = self.to_seq(compressed)
        z0 = seq_feat.reshape(seq_feat.shape[0], self.intermediate_chans, 1, self.num_trajectory_steps)
        z1 = torch.relu(self.temporal_conv1(z0))
        z2 = self.temporal_conv2(z1)
        z3 = z0 + z2
        trajectory = self.out_conv(z3)
        return trajectory.permute(0, 3, 2, 1).squeeze(2)
