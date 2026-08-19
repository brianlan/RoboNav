import torch
from mmengine.dist import all_reduce
from mmengine.runner import ValLoop
from mmengine.runner.amp import autocast
from prefusion.runner.loops import StreamingSequenceBatchTrainLoop

from robonav.registry import LOOPS

__all__ = ["StreamingSequenceBPTTTrainLoop", "AquaSequenceValLoop"]


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


@LOOPS.register_module()
class AquaSequenceValLoop(ValLoop):
    """Validation over ``SequenceBatchDataset`` sequence batches.

    Each dataloader item is one sequence batch (a list of frame
    batches): the recurrent model is reset once per sequence batch,
    then every frame is processed in order through the standard
    ``val_step`` (mode="predict") path with one ``run_iter`` (and its
    hooks) per frame, using flattened indices like Prefusion's
    ``SequenceBatchValLoop``. Each rank owns its SRU state; states are
    never synchronized. Scheme A: every sample actually produced by
    the dataset/sampler counts as valid, including duplicated
    sequence slots and round_up padding, so the evaluator receives
    the all-reduced global number of processed samples and losses are
    averaged weighted by each frame batch's sample count.
    """

    def run(self) -> dict:
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        self._loss_sums = {}
        self._num_samples = 0
        sequence_length = self.dataloader.dataset.sequence_length
        model = self.runner.model
        for sequence_idx, sequence_batch in enumerate(self.dataloader):
            (model.module if hasattr(model, "module") else model).reset()
            for frame_idx, frame_batch in enumerate(sequence_batch):
                self.run_iter(sequence_idx * sequence_length + frame_idx, frame_batch)
        loss_metrics, global_samples = self._reduce_stats()
        metrics = self.evaluator.evaluate(global_samples)
        metrics.update(loss_metrics)
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, frame_batch):
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=frame_batch)
        with autocast(enabled=self.fp16):
            *predictions, loss_element = self.runner.model.val_step(frame_batch)
        batch = len(predictions)
        for name, value in loss_element.loss.items():
            # accumulate as device tensors; only the final all-reduced
            # scalars are converted to Python values
            self._loss_sums[name] = (
                self._loss_sums.get(name, 0.0) + value.detach() * batch
            )
        self._num_samples += batch
        self.evaluator.process(data_samples=predictions, data_batch=frame_batch)
        self.runner.call_hook(
            "after_val_iter", batch_idx=idx, data_batch=frame_batch, outputs=predictions
        )

    def _reduce_stats(self):
        """All-reduce per-component loss sums and the sample count across
        ranks; return globally averaged loss_* components (+ ``val_loss``
        as their sum) and the global sample count."""
        names = sorted(self._loss_sums)
        sums = torch.stack([self._loss_sums[name] for name in names])
        stats = torch.cat((sums, sums.new_tensor(self._num_samples).unsqueeze(0)))
        stats = stats.double()
        all_reduce(stats)
        count = stats[-1].item()
        metrics = {name: stats[i].item() / count for i, name in enumerate(names)}
        metrics["val_loss"] = sum(metrics.values())
        return metrics, int(count)
