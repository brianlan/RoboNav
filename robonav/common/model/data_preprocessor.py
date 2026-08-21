from typing import Any

import torch
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from robonav.registry import MODELS


__all__ = ["FrameBatchMerger"]


@MODELS.register_module()
class FrameBatchMerger(BaseDataPreprocessor):
    def __init__(self, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self._device = device

    def forward(
        self, data: list[list[dict[str, Any]]] | list[dict[str, Any]],
        training: bool = False,
    ) -> dict[str, Any]:
        if training:
            # Training data is one sequence of frame batches: merge every
            # frame separately and return an explicit sequence keyword for
            # the model's full-sequence BPTT forward.
            return {"sequence": [self._merge_frame(frame) for frame in data]}
        return self._merge_frame(data)

    def _merge_frame(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        merged = {}
        for key in data[0].keys():
            merged[key] = [self._cast_data(i[key]) for i in data]
        return merged

    def _cast_data(self, data: Any):
        if isinstance(data, torch.Tensor):
            _dtype = torch.float32 if "float" in str(data.dtype) else data.dtype
            return data.to(dtype=_dtype, device=self._device)
        if isinstance(data, dict):
            return {k: self._cast_data(data[k]) for k in data}
        if isinstance(data, list):
            return [self._cast_data(d) for d in data]
        return data
