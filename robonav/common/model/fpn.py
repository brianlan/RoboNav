from torchvision.ops import FeaturePyramidNetwork

from robonav.common.registry import MODELS

MODELS.register_module(name='TvFPN', module=FeaturePyramidNetwork)
