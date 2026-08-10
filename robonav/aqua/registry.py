from mmengine import Registry

from prefusion.registry import MODELS as PREFUSION_MODELS

MODELS = Registry(
    "aqua",
    parent=PREFUSION_MODELS,
    locations=["robonav.aqua"],
    scope="aqua",
)
