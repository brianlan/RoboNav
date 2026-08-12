from mmengine import Registry
from prefusion.registry import (
    METRICS as PREFUSION_METRICS,
    MODELS as PREFUSION_MODELS,
    MODEL_FEEDERS as PREFUSION_MODEL_FEEDERS,
    TENSOR_SMITHS as PREFUSION_TENSOR_SMITHS,
)

MODELS = Registry("model", parent=PREFUSION_MODELS, scope="robonav")
MODEL_FEEDERS = Registry(
    "model_feeder", parent=PREFUSION_MODEL_FEEDERS, scope="robonav"
)
TENSOR_SMITHS = Registry(
    "tensor_smith", parent=PREFUSION_TENSOR_SMITHS, scope="robonav"
)
METRICS = Registry("metric", parent=PREFUSION_METRICS, scope="robonav")

__all__ = ["MODELS", "MODEL_FEEDERS", "TENSOR_SMITHS", "METRICS"]
