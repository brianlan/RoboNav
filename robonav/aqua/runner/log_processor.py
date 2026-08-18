from prefusion.hooks.log_processor import SequenceAwareLogProcessor

from robonav.registry import LOG_PROCESSORS

__all__ = ["StreamingSequenceBPTTLogProcessor"]


@LOG_PROCESSORS.register_module()
class StreamingSequenceBPTTLogProcessor(SequenceAwareLogProcessor):
    def _get_dataloader_size(self, runner, mode) -> int:
        if mode == "train":
            return len(runner.train_loop.dataloader)
        return super()._get_dataloader_size(runner, mode)
