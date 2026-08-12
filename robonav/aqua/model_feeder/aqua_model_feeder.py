from prefusion.dataset.model_feeder import BaseModelFeeder

from ..registry import MODEL_FEEDERS


__all__ = ["AquaModelFeeder"]


class AquaModelFeeder(BaseModelFeeder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, frame_batch):
        a = 100
        pass
