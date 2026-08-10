import warnings

import timm
from mmpretrain.models.backbones.timm_backbone import print_timm_feature_info
from mmpretrain.utils import require

from robonav.common.registry import MODELS
from prefusion.models import BaseModel

__all__ = ["TIMMBackbone"]


@MODELS.register_module()
class TIMMBackbone(BaseModel):
    @require("timm")
    def __init__(
        self,
        model_name: str,
        features_only: bool = False,
        pretrained: bool = False,
        checkpoint_path: str = "",
        in_channels: int = 3,
        freeze: bool = False,
        fixbn: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be bool, not str for model path")
        if features_only and checkpoint_path:
            warnings.warn("Using both features_only and checkpoint_path will cause an error in timm")

        super().__init__(freeze=freeze, fixbn=fixbn, init_cfg=init_cfg)
        if "norm_layer" in kwargs:
            norm_class = MODELS.get(kwargs["norm_layer"])

            def build_norm(*args, **kwargs):
                return norm_class(*args, **kwargs)

            kwargs["norm_layer"] = build_norm

        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        if hasattr(self.timm_model, "reset_classifier"):
            self.timm_model.reset_classifier(0, "")

        if pretrained or checkpoint_path:
            self._is_init = True

        print_timm_feature_info(getattr(self.timm_model, "feature_info", None))

    def forward(self, x):
        features = self.timm_model(x)
        if isinstance(features, (list, tuple)):
            return tuple(features)
        return (features,)
