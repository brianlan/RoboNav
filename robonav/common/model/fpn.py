from torchvision.ops import FeaturePyramidNetwork

from robonav.registry import MODELS

MODELS.register_module(name='TvFPN', module=FeaturePyramidNetwork)
