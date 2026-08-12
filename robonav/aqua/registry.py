from mmengine import Registry

from robonav.common.registry import MODELS as ROBONAV_MODELS
from robonav.common.registry import MODEL_FEEDERS as ROBONAV_MODEL_FEEDERS
from robonav.common.registry import TENSOR_SMITHS as ROBONAV_TENSOR_SMITHS

MODELS = Registry(
    "model",
    parent=ROBONAV_MODELS,
    locations=["robonav.aqua"],
    scope="aqua",
)

MODEL_FEEDERS = Registry(
    "model_feeder", 
    parent=ROBONAV_MODEL_FEEDERS, 
    locations=["robonav.aqua"], 
    scope="aqua"
)

TENSOR_SMITHS = Registry(
    "tensor_smith", 
    parent=ROBONAV_TENSOR_SMITHS, 
    locations=["robonav.aqua"], 
    scope="aqua"
)
