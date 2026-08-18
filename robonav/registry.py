from mmengine import Registry
from prefusion.registry import (
    LOOPS as PREFUSION_LOOPS,
    METRICS as PREFUSION_METRICS,
    MODELS as PREFUSION_MODELS,
    MODEL_FEEDERS as PREFUSION_MODEL_FEEDERS,
    TENSOR_SMITHS as PREFUSION_TENSOR_SMITHS,
    TRANSFORMABLES as PREFUSION_TRANSFORMABLES,
    TRANSFORMABLE_LOADERS as PREFUSION_TRANSFORMABLE_LOADERS,
)

LOOPS = Registry("loop", parent=PREFUSION_LOOPS, scope="robonav")
MODELS = Registry("model", parent=PREFUSION_MODELS, scope="robonav")
MODEL_FEEDERS = Registry(
    "model_feeder", parent=PREFUSION_MODEL_FEEDERS, scope="robonav"
)
TENSOR_SMITHS = Registry(
    "tensor_smith", parent=PREFUSION_TENSOR_SMITHS, scope="robonav"
)
METRICS = Registry("metric", parent=PREFUSION_METRICS, scope="robonav")
TRANSFORMABLES = Registry(
    "transformable", parent=PREFUSION_TRANSFORMABLES, scope="robonav"
)
TRANSFORMABLE_LOADERS = Registry(
    "transformable_loader", parent=PREFUSION_TRANSFORMABLE_LOADERS, scope="robonav"
)

__all__ = [
    "LOOPS",
    "MODELS",
    "MODEL_FEEDERS",
    "TENSOR_SMITHS",
    "METRICS",
    "TRANSFORMABLES",
    "TRANSFORMABLE_LOADERS",
]
