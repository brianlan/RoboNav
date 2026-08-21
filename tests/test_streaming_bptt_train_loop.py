import types

import pytest
import torch
from mmengine.model import MMDistributedDataParallel
from mmengine.optim import OptimWrapper

from robonav.aqua.model.aqua import AquaNet
from robonav.aqua.runner.loop import StreamingSequenceBPTTTrainLoop
from robonav.common.model.data_preprocessor import FrameBatchMerger


class FakeFuser(torch.nn.Module):
    """Minimal recurrent core with a differentiable, resettable state."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.state = None
        self.frame = 0
        self.state_at_entry = []
        self.state_grads = []
        self.states_out = []

    def reset(self):
        self.state = None
        self.frame = 0

    def forward(self, x):
        self.state_at_entry.append(self.state)
        prev = torch.zeros_like(x) if self.state is None else self.state
        state = prev * self.weight + x
        frame = self.frame
        state.register_hook(lambda g: self.state_grads.append((frame, g)))
        self.state = state
        self.states_out.append(state)
        self.frame += 1
        return state


def _frame(x):
    """One frame microbatch: a single-sample dict with tensor ``x``."""
    return [{"x": x}]


class FakeBPTTModel(AquaNet):
    """AquaNet stand-in reusing the real sequence forward workflow through
    the default ``BaseModel.train_step`` and ``FrameBatchMerger``."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.data_preprocessor = FrameBatchMerger(device="cpu")
        self.temporal_fuser = FakeFuser()
        self.frames_seen = []

    def _forward_frame(self, x=None, mode="loss", **kwargs):
        self.frames_seen.append(x[0])
        first_frame = self.temporal_fuser.frame == 0
        state = self.temporal_fuser(x[0])
        # First frame's loss is scaled to zero, so its only gradient path is
        # through the recurrent state into later frames' losses.
        return dict(loss_main=state.sum() * (0.0 if first_frame else 1.0))


class FakeEarlyParamModel(FakeBPTTModel):
    """Adds a parameter that only the first frame of a sequence uses."""

    def __init__(self):
        super().__init__()
        self.early = torch.nn.Parameter(torch.tensor(0.5))

    def _forward_frame(self, x=None, mode="loss", **kwargs):
        losses = super()._forward_frame(x=x, mode=mode)
        if self.temporal_fuser.frame == 1:  # the first frame just ran
            losses["loss_main"] = losses["loss_main"] + self.early * x[0].sum()
        return losses


class FakeMultiLossModel(AquaNet):
    """AquaNet stand-in whose per-frame loss dict has multiple loss_* keys
    plus a non-loss metric, to exercise the default train_step parsing of
    time-averaged sequence outputs."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.data_preprocessor = FrameBatchMerger(device="cpu")
        self.temporal_fuser = FakeFuser()

    def _forward_frame(self, x=None, mode="loss", **kwargs):
        t = self.temporal_fuser.frame
        state_sum = self.temporal_fuser(x[0]).sum()
        return dict(
            loss_main=state_sum,
            loss_aux=10.0 * state_sum,
            acc=torch.tensor(float(t + 1)),
        )


class CountingOptimWrapper(OptimWrapper):
    def __init__(self, model):
        super().__init__(torch.optim.SGD(model.parameters(), lr=0.1))
        self._model = model
        self.step_frames = []

    def step(self, *args, **kwargs):
        self.step_frames.append(len(self._model.frames_seen))
        super().step(*args, **kwargs)


class CapturingOptimWrapper(OptimWrapper):
    """Records the scalar objective tensor handed to update_params."""

    def __init__(self, model):
        super().__init__(torch.optim.SGD(model.parameters(), lr=0.1))
        self.objectives = []

    def update_params(self, loss, **kwargs):
        self.objectives.append(loss.detach().clone())
        super().update_params(loss, **kwargs)


class FakeRunner:
    def __init__(self, model, optim_wrapper):
        self.model = model
        self.optim_wrapper = optim_wrapper
        self.visualizer = types.SimpleNamespace()
        self.val_loop = None
        self.hooks = []

    def call_hook(self, name, **kwargs):
        self.hooks.append((name, kwargs.get("batch_idx")))


class FakeDataset:
    metainfo = {}

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length


class FakeDataLoader:
    def __init__(self, sequences, sequence_length):
        self.sequences = sequences
        self.dataset = FakeDataset(sequence_length)

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return iter([frame for seq in self.sequences for frame in seq])


# ---------------------------------------------------------------------------
# FrameBatchMerger: frame regression and nested training sequence
# ---------------------------------------------------------------------------


def test_frame_batch_merger_merges_single_frame_batch():
    merger = FrameBatchMerger(device="cpu")
    frame = [
        {"x": torch.ones(2), "meta": 1},
        {"x": torch.zeros(2), "meta": 2},
    ]
    merged = merger(frame, False)
    assert set(merged) == {"x", "meta"}
    torch.testing.assert_close(merged["x"][0], torch.ones(2))
    torch.testing.assert_close(merged["x"][1], torch.zeros(2))
    assert merged["meta"] == [1, 2]
    # float64 inputs are cast to float32 like any other float dtype
    assert merger([{"x": torch.ones(1, dtype=torch.float64)}], False)["x"][
        0
    ].dtype == torch.float32


def test_frame_batch_merger_training_merges_nested_sequence():
    merger = FrameBatchMerger(device="cpu")
    sequence = [
        [{"x": torch.ones(2)}, {"x": 2.0 * torch.ones(2)}],
        [{"x": torch.zeros(2)}, {"x": 3.0 * torch.ones(2)}],
    ]
    out = merger(sequence, True)
    # explicit sequence keyword dict, one independently merged dict per frame
    assert set(out) == {"sequence"}
    assert len(out["sequence"]) == 2
    torch.testing.assert_close(out["sequence"][0]["x"][0], torch.ones(2))
    torch.testing.assert_close(out["sequence"][0]["x"][1], 2.0 * torch.ones(2))
    torch.testing.assert_close(out["sequence"][1]["x"][0], torch.zeros(2))
    torch.testing.assert_close(out["sequence"][1]["x"][1], 3.0 * torch.ones(2))


# ---------------------------------------------------------------------------
# Single-GPU default BaseModel.train_step over full sequences
# ---------------------------------------------------------------------------


def test_bptt_loop_end_to_end():
    T = 3
    # Second sequence deliberately has a different batch size.
    sequences = [
        [_frame(torch.randn(2, 4)) for _ in range(T)],
        [_frame(torch.randn(3, 4)) for _ in range(T)],
    ]
    dataloader = FakeDataLoader(sequences, sequence_length=T)
    model = FakeBPTTModel()
    wrapper = CountingOptimWrapper(model)
    runner = FakeRunner(model, wrapper)

    loop = StreamingSequenceBPTTTrainLoop(runner, dataloader, max_epochs=1, val_interval=-1)
    # len(dataloader) counts sequence batches, not frame microbatches.
    assert loop.max_iters == 2
    loop.run_epoch()

    # Exactly one optimizer step per sequence, only after all T forwards
    # (frames_seen is cumulative across the epoch).
    assert wrapper.step_frames == [T, 2 * T]
    assert len(model.frames_seen) == 2 * T

    # Iteration count and hooks are sequence-level.
    assert loop.iter == 2
    hook_names = [name for name, _ in runner.hooks]
    assert hook_names.count("before_train_iter") == 2
    assert hook_names.count("after_train_iter") == 2
    assert [idx for name, idx in runner.hooks if name == "before_train_iter"] == [0, 1]

    fuser = model.temporal_fuser
    # Each sequence starts from a reset state; state is cleared after the
    # final forward so nothing (including another batch size) is inherited.
    assert [s is None for s in fuser.state_at_entry] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert fuser.state is None

    # Every frame state received a finite gradient; frame 0's only gradient
    # path is through the recurrent state, proving early frames contribute.
    assert len(fuser.state_grads) == 2 * T
    frame0_grads = [g for i, g in fuser.state_grads if i == 0]
    assert len(frame0_grads) == 2
    for g in frame0_grads:
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_train_step_averages_multiple_loss_keys():
    T = 4
    model = FakeMultiLossModel()
    wrapper = CapturingOptimWrapper(model)
    frames = [_frame(torch.randn(2, 3)) for _ in range(T)]

    log_vars = model.train_step(frames, wrapper)

    # Exactly one optimizer update for the whole sequence.
    assert len(wrapper.objectives) == 1
    state_sums = [s.sum() for s in model.temporal_fuser.states_out]
    # default parse_losses sums every loss_* key of the time-averaged dict:
    # mean_t(loss_main + loss_aux) == mean_t(loss_main) + mean_t(loss_aux)
    expected = torch.stack([s + 10.0 * s for s in state_sums]).mean()
    torch.testing.assert_close(wrapper.objectives[0], expected)

    # Returned log vars are averaged over time; the non-loss key is retained.
    torch.testing.assert_close(float(log_vars["loss"]), float(expected))
    torch.testing.assert_close(
        float(log_vars["loss_aux"]), float(torch.stack([10.0 * s for s in state_sums]).mean())
    )
    assert float(log_vars["acc"]) == (T + 1) / 2


def test_loop_registered_in_robonav_loops():
    import robonav  # noqa: F401
    from prefusion.registry import LOOPS as PREFUSION_LOOPS
    from robonav.registry import LOOPS

    assert LOOPS.get("StreamingSequenceBPTTTrainLoop") is StreamingSequenceBPTTTrainLoop
    assert (
        PREFUSION_LOOPS.get("robonav.StreamingSequenceBPTTTrainLoop")
        is StreamingSequenceBPTTTrainLoop
    )


def test_loop_uses_dataset_sequence_length():
    T = 5  # not 20: grouping must come from dataset.sequence_length
    dataloader = FakeDataLoader(
        [[_frame(torch.randn(1, 4)) for _ in range(T)]], sequence_length=T
    )
    model = FakeBPTTModel()
    wrapper = CountingOptimWrapper(model)
    loop = StreamingSequenceBPTTTrainLoop(
        FakeRunner(model, wrapper), dataloader, max_epochs=1, val_interval=-1
    )
    loop.run_epoch()
    assert wrapper.step_frames == [T]
    assert len(model.frames_seen) == T


# ---------------------------------------------------------------------------
# Distributed default MMDistributedDataParallel.train_step (2-process gloo)
# ---------------------------------------------------------------------------


def _rank_frames(rank, T=3):
    """Deterministic per-rank sequence, known to every rank and the main
    process so a global-batch reference can be computed locally."""
    return [_frame(torch.full((2, 3), float(10 * rank + t + 1))) for t in range(T)]


def _ddp_worker(rank, world_size, init_file, queue, find_unused):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    model = FakeEarlyParamModel() if find_unused else FakeBPTTModel()
    ddp = MMDistributedDataParallel(model, find_unused_parameters=find_unused)
    wrapper = OptimWrapper(torch.optim.SGD(model.parameters(), lr=0.1))

    root_forwards = []
    ddp_forward = ddp.forward

    def counting_forward(*args, **kwargs):
        root_forwards.append(1)
        return ddp_forward(*args, **kwargs)

    ddp.forward = counting_forward
    ddp.train_step(_rank_frames(rank), wrapper)
    dist.destroy_process_group()
    # plain floats: tensors would be shared through file descriptors that
    # close when the worker exits
    queue.put(
        (
            {name: value.item() for name, value in model.named_parameters()},
            root_forwards,
            len(model.frames_seen),
        )
    )


def _global_batch_reference(find_unused, world_size=2, lr=0.1):
    """Replay every rank's sequence on one model, then take a single SGD
    step on the mean loss: DDP's gradient averaging must reproduce it."""
    model = FakeEarlyParamModel() if find_unused else FakeBPTTModel()
    losses = [
        model(**model.data_preprocessor(_rank_frames(rank), training=True), mode="loss")
        for rank in range(world_size)
    ]
    total = sum(
        sum(value for key, value in loss.items() if "loss" in key) for loss in losses
    ) / world_size
    total.backward()
    torch.optim.SGD(model.parameters(), lr=lr).step()
    return {name: value.detach().clone() for name, value in model.named_parameters()}


@pytest.mark.parametrize("find_unused", [False, True])
def test_default_ddp_train_step_matches_global_batch(find_unused, tmp_path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    init_file = str(tmp_path / "dist_init")
    procs = [
        ctx.Process(target=_ddp_worker, args=(rank, 2, init_file, queue, find_unused))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    assert all(proc.exitcode == 0 for proc in procs)

    results = [queue.get() for _ in range(2)]
    reference = _global_batch_reference(find_unused)
    for params, root_forwards, frames in results:
        # the whole sequence crossed the root DDP forward exactly once,
        # with every frame forwarded inside it
        assert root_forwards == [1]
        assert frames == 3
        assert set(params) == set(reference)
        for name, value in params.items():
            assert value == pytest.approx(reference[name].item())
    # both ranks hold identical parameters after the synchronized step
    assert (
        results[0][0]["temporal_fuser.weight"]
        == results[1][0]["temporal_fuser.weight"]
    )
    if find_unused:
        # a parameter used only by the first frame still received gradient
        assert results[0][0]["early"] != 0.5
