from prefusion.runner.loops import StreamingSequenceBatchTrainLoop

from robonav.registry import LOOPS

__all__ = ["StreamingSequenceBPTTTrainLoop"]


@LOOPS.register_module()
class StreamingSequenceBPTTTrainLoop(StreamingSequenceBatchTrainLoop):
    """Streaming full-sequence BPTT training.

    Unlike ``StreamingSequenceBatchTrainLoop`` (one optimizer step per frame
    microbatch), one loop iteration consumes one whole sequence of aligned
    frame microbatches and calls ``model.train_step`` once with the nested
    sequence, so the model can backpropagate through time.
    """

    def __init__(
        self,
        runner,
        dataloader,
        max_epochs,
        val_begin=1,
        val_interval=1,
        dynamic_intervals=None,
    ):
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        # AlignedTimestepBatchSampler.__len__ counts sequence batches, so one
        # iteration here is one sequence, not one frame microbatch.
        self._max_iters = self._max_epochs * len(self.dataloader)

    def run_epoch(self) -> None:
        """Iterate one epoch, grouping streamed frames into full sequences."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()

        if hasattr(self.dataloader, "batch_sampler") and hasattr(self.dataloader.batch_sampler, "set_epoch"):
            self.dataloader.batch_sampler.set_epoch(self.epoch)
        if hasattr(self.dataloader, "dataset") and hasattr(self.dataloader.dataset, "set_epoch"):
            self.dataloader.dataset.set_epoch(self.epoch)

        sequence_length = self.dataloader.dataset.sequence_length
        frames = iter(self.dataloader)
        for seq_idx in range(len(self.dataloader)):
            sequence = [next(frames) for _ in range(sequence_length)]
            self.run_iter(seq_idx, sequence)

        if hasattr(self.dataloader.dataset, "post_epoch_processing"):
            self.dataloader.dataset.post_epoch_processing()

        self.runner.call_hook("after_train_epoch")
        self._epoch += 1
