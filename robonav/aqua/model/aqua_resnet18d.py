import timm
import torch
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

        self.pe_encoder = self._make_pe_encoder(6, 64)
        self.rgb_pe_fuse = self._make_rgb_pe_fuse(64, 64, 64)

        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

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

    @staticmethod
    def _make_pe_encoder(in_chan, out_chan):
        return nn.Conv2d(in_chan, out_chan, 1)

    @staticmethod
    def _make_rgb_pe_fuse(rgb_chan, pe_chan, out_chan):
        return nn.Conv2d(rgb_chan + pe_chan, out_chan, 1)

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

    def forward(self, rgb, pe, goal, state, hidden):
        rgb = self.conv1(rgb)
        rgb = self.bn1(rgb)
        rgb = self.act1(rgb)

        # fuse rgb and pe
        pe = self.pe_encoder(pe)
        rgb_w_pe = torch.cat((rgb, pe), dim=1)
        rgb_w_pe = self.rgb_pe_fuse(rgb_w_pe)
        rgb_w_pe = self.maxpool(rgb_w_pe)

        x1 = self.layer1(rgb_w_pe)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, hidden
