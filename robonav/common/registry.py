from mmengine import Registry

from prefusion.registry import MODELS as PREFUSION_MODELS

MODELS = Registry(
    "robonav",
    parent=PREFUSION_MODELS,
    locations=["robonav.common"],
    scope="common",
)
