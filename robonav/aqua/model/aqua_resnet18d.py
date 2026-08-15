import timm
import torch
import torch.nn.functional as F
from torch import nn

from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["AquaResNet18D"]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop_block = nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.aa = nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act2(x + shortcut)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


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


class SpatialFeatureCompressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat):
        return feat


class SRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden = None
        self.cell = None

    def forward(self, input):
        return input


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


@MODELS.register_module()
class AquaResNet18D(BaseModel):
    model_name = "resnet18d.ra4_e3600_r224_in1k"

    def __init__(
        self,
        model_name=model_name,
        features_only=True,
        pretrained=False,
        out_indices=(1, 2, 3, 4),
        in_channels=3,
        freeze=False,
        fixbn=False,
        init_cfg=None,
    ):
        if (
            model_name != self.model_name
            or not features_only
            or tuple(out_indices) != (1, 2, 3, 4)
        ):
            raise ValueError(
                "AquaResNet18D only supports resnet18d.ra4_e3600_r224_in1k with "
                "features_only=True and out_indices=(1, 2, 3, 4)"
            )

        super().__init__(freeze=freeze, fixbn=fixbn, init_cfg=init_cfg)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.pe_encoder = nn.Conv2d(6, 64, 1)
        self.rgb_pe_fuse = nn.Conv2d(64 + 64, 64, 1)

        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.film_by_goal_n_state = FiLMByGoalAndState()
        self.spatial_gate = SpatialGate()
        self.vfeat_compressor = SpatialFeatureCompressor()
        self.f4_conv_1x1 = nn.Conv2d(512, 256, 1)
        self.f3_conv_1x1 = nn.Conv2d(256, 256, 1)
        self.sru = SRU()
        self.temporal_film = TemporalFiLM()
        self.history_enhanced_compressor = SpatialFeatureCompressor()

        self._init_weights()
        if pretrained:
            source = timm.create_model(
                self.model_name,
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=in_channels,
            )
            self.load_state_dict(source.state_dict(), strict=False)
            self._is_init = True

    @staticmethod
    def _make_layer(in_channels, out_channels, stride=1):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, BasicBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, rgb, pe, goal, ego_poses):
        device = rgb.device
        rgb = self.conv1(rgb)
        rgb = self.bn1(rgb)
        rgb = self.act1(rgb)

        # fuse rgb and pe
        pe = self.pe_encoder(pe)
        rgb_w_pe = torch.cat((rgb, pe), dim=1)
        rgb_w_pe = self.rgb_pe_fuse(rgb_w_pe)
        rgb_w_pe = self.maxpool(rgb_w_pe)

        f1 = self.layer1(rgb_w_pe)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        cur_velo = self._get_cur_velocity(ego_poses, device)
        delta_pose = self._calc_delta_pose(ego_poses)

        f4m = self.film_by_goal_n_state(f4, goal, cur_velo)
        f4m_up = F.interpolate(self.f4_conv_1x1(f4m), scale_factor=2, mode="nearest")
        f3_fused = F.relu(f4m_up + self.f3_conv_1x1(f3))
        f3g = self.spatial_gate(f3_fused, goal, cur_velo)
        scene_desc = self.vfeat_compressor(f3g)
        cur_state = torch.cat((scene_desc, cur_velo, delta_pose, goal), dim=1)
        hidden = self.sru(cur_state)
        f3g_history_injected = self.temporal_film(f3g, hidden)
        final_feat = self.history_enhanced_compressor(f3g_history_injected)

        return f1, f2, f3, f4, final_feat, hidden

    @staticmethod
    def _get_cur_velocity(ego_poses, device):
        return torch.vstack(
            [e.transformables["0"].linear_velocity for e in ego_poses]
        ).to(device=device)

    @staticmethod
    def _calc_delta_pose(ego_poses, device):
        pass

    def construct_sru_state(self, goal, ego_poses):
        pass
