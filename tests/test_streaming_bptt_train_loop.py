import types

import torch
from mmengine.optim import OptimWrapper

from robonav.aqua.model.aqua import AquaNet
from robonav.aqua.runner.loop import StreamingSequenceBPTTTrainLoop


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


class FakeBPTTModel(AquaNet):
    """AquaNet stand-in reusing the real sequence train_step, tiny core."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.data_preprocessor = lambda data, training=False: data
        self.temporal_fuser = FakeFuser()
        self.frames_seen = []

    def forward(self, *, x, mode="loss", **kwargs):
        self.frames_seen.append(x)
        first_frame = self.temporal_fuser.frame == 0
        state = self.temporal_fuser(x)
        # First frame's loss is scaled to zero, so its only gradient path is
        # through the recurrent state into later frames' losses.
        return dict(loss=state.sum() * (0.0 if first_frame else 1.0))


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


def test_bptt_loop_end_to_end():
    T = 3
    # Second sequence deliberately has a different batch size.
    sequences = [
        [{"x": torch.randn(2, 4)} for _ in range(T)],
        [{"x": torch.randn(3, 4)} for _ in range(T)],
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
    # final backward so nothing (including another batch size) is inherited.
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


class FakeMultiLossModel(AquaNet):
    """AquaNet stand-in whose per-frame loss dict has multiple loss keys
    plus a non-loss key, to exercise parse_losses-based aggregation."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.data_preprocessor = lambda data, training=False: data
        self.temporal_fuser = FakeFuser()
        self.frame_idx = 0

    def forward(self, *, x, mode="loss", **kwargs):
        t = self.frame_idx
        self.frame_idx += 1
        state_sum = self.temporal_fuser(x).sum()
        return dict(loss=state_sum, aux_loss=10.0 * state_sum, acc=torch.tensor(float(t + 1)))


def test_train_step_averages_multiple_loss_keys():
    T = 4
    model = FakeMultiLossModel()
    wrapper = CapturingOptimWrapper(model)
    frames = [{"x": torch.randn(2, 3)} for _ in range(T)]

    log_vars = model.train_step(frames, wrapper)

    # Exactly one optimizer update for the whole sequence.
    assert len(wrapper.objectives) == 1
    state_sums = [s.sum() for s in model.temporal_fuser.states_out]
    # parse_losses sums every key containing "loss": loss + aux_loss per frame.
    expected = torch.stack([s + 10.0 * s for s in state_sums]).mean()
    torch.testing.assert_close(wrapper.objectives[0], expected)

    # Returned log vars are averaged over time; the non-loss key is retained.
    torch.testing.assert_close(float(log_vars["loss"]), float(expected))
    torch.testing.assert_close(
        float(log_vars["aux_loss"]), float(torch.stack([10.0 * s for s in state_sums]).mean())
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
    dataloader = FakeDataLoader([[{"x": torch.randn(1, 4)} for _ in range(T)]], sequence_length=T)
    model = FakeBPTTModel()
    wrapper = CountingOptimWrapper(model)
    loop = StreamingSequenceBPTTTrainLoop(
        FakeRunner(model, wrapper), dataloader, max_epochs=1, val_interval=-1
    )
    loop.run_epoch()
    assert wrapper.step_frames == [T]
    assert len(model.frames_seen) == T
