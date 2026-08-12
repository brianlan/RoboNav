from mmengine import Registry

from mmengine.registry import MODELS as MMENGINE_MODELS
from prefusion.registry import MODEL_FEEDERS as PREFUSION_MODEL_FEEDERS
from prefusion.registry import TENSOR_SMITHS as PREFUSION_TENSOR_SMITHS

MODELS = Registry(
    "model",
    parent=MMENGINE_MODELS,
    locations=["robonav.common"],
    scope="robonav",
)

MODEL_FEEDERS = Registry(
    "model_feeder", 
    parent=PREFUSION_MODEL_FEEDERS, 
    locations=["robonav.common"], 
    scope="robonav"
)

TENSOR_SMITHS = Registry(
    "tensor_smith", 
    parent=PREFUSION_TENSOR_SMITHS, 
    locations=["robonav.common"], 
    scope="robonav"
)

a=10