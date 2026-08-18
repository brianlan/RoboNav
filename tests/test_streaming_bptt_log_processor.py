import os
import types

from mmengine.config import Config
from prefusion.registry import LOG_PROCESSORS as PREFUSION_LOG_PROCESSORS

from robonav.aqua.runner.log_processor import StreamingSequenceBPTTLogProcessor
from robonav.registry import LOG_PROCESSORS


class _Dataset:
    sequence_length = 20


class _DataLoader:
    dataset = _Dataset()

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_sequence_batch_train_size_and_sequence_aware_eval_sizes(monkeypatch):
    train_loop = types.SimpleNamespace(dataloader=_DataLoader(5))
    runner = types.SimpleNamespace(
        train_loop=train_loop,
        val_loop=types.SimpleNamespace(dataloader=_DataLoader(2)),
        test_loop=types.SimpleNamespace(dataloader=_DataLoader(3)),
        _train_loop=train_loop,
        max_epochs=2,
        epoch=0,
        iter=0,
        message_hub=types.SimpleNamespace(runtime_info={}),
    )
    processor = StreamingSequenceBPTTLogProcessor(tabulate=False)

    assert processor._get_dataloader_size(runner, "train") == 5
    assert processor._get_dataloader_size(runner, "val") == 40
    assert processor._get_dataloader_size(runner, "test") == 60

    monkeypatch.setattr(processor, "_parse_windows_size", lambda *args: [])
    monkeypatch.setattr(processor, "_collect_scalars", lambda *args: {})
    monkeypatch.setattr("prefusion.hooks.log_processor.is_cuda_available", lambda: False)
    monkeypatch.setattr("prefusion.hooks.log_processor.is_musa_available", lambda: False)
    _, log = processor.get_log_after_iter(runner, batch_idx=4, mode="train")
    assert "Epoch(train)" in log
    assert "[5/5]" in log


def test_log_processor_registry_and_config_build():
    config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__), "..", "configs", "kinogoal_dla_resnet18_overfit.py"
        )
    )

    assert LOG_PROCESSORS.get("StreamingSequenceBPTTLogProcessor") is StreamingSequenceBPTTLogProcessor
    assert (
        PREFUSION_LOG_PROCESSORS.get("robonav.StreamingSequenceBPTTLogProcessor")
        is StreamingSequenceBPTTLogProcessor
    )
    processor = PREFUSION_LOG_PROCESSORS.build(config.log_processor)
    assert isinstance(processor, StreamingSequenceBPTTLogProcessor)
    assert processor.tabulate_ncols == 5
