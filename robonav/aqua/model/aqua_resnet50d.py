import timm
import torch
from torch import nn

from prefusion.models import BaseModel

from robonav.registry import MODELS

__all__ = ["AquaResNet50D"]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.drop_block = nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.aa = nn.Identity()
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False)
                if stride != 1
                else nn.Identity(),
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
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act3(x + shortcut)


@MODELS.register_module()
class AquaResNet50D(BaseModel):
    model_name = "resnet50d.ra4_e3600_r224_in1k"

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
                "AquaResNet50D only supports resnet50d.ra4_e3600_r224_in1k with "
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

        self.layer1 = self._make_layer(64, 256, num_blocks=3)
        self.layer2 = self._make_layer(256, 512, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 1024, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, num_blocks=3, stride=2)

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
    def _make_layer(in_channels, out_channels, num_blocks, stride=1):
        return nn.Sequential(
            Bottleneck(in_channels, out_channels, stride),
            *(Bottleneck(out_channels, out_channels) for _ in range(num_blocks - 1)),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, Bottleneck):
                nn.init.zeros_(module.bn3.weight)

    def forward(self, rgb, pe):
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

        return f1, f2, f3, f4
