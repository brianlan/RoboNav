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
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        training: bool = False,
    ) -> dict[str, Any]:
        if isinstance(data, dict) and "sequence" in data:
            # Explicit sequence envelope from StreamingSequenceBPTTTrainLoop:
            # merge every frame separately and return an explicit sequence
            # keyword for the model's full-sequence BPTT forward. Dispatch is
            # by input field, never by the training flag.
            return {"sequence": [self._merge_frame(frame) for frame in data["sequence"]]}
        return self._merge_frame(data)

    def _merge_frame(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[Any] | bool]:
        merged = {}
        for key in data[0].keys():
            merged[key] = [self._cast_data(i[key]) for i in data]
        # Adapter seam: one batch-level recurrent-state boundary from the
        # Prefusion occurrences. Every occurrence must carry a real bool
        # (None means unassigned and must not be coerced) and all samples of
        # a time-aligned frame batch must agree; None or mixed values would
        # silently corrupt recurrent state. A missing index_info key fails
        # naturally at the required lookup below.
        starts = [index_info.stream_start for index_info in merged["index_info"]]
        if not all(isinstance(start, bool) for start in starts) or len(set(starts)) != 1:
            raise ValueError(
                "stream_start must agree across a recurrent frame batch and be a real bool, got "
                f"{starts}"
            )
        merged["stream_start"] = starts[0]
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
