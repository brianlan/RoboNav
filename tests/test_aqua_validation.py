"""Focused tests for Scheme A training-time validation of AquaNet."""

import math
import os
import types

import pytest
import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.structures import BaseDataElement
from prefusion.dataset.index_info import IndexInfo

from robonav.aqua.metric import AquaTrajectoryMetric, frame_trajectory_metrics
from robonav.aqua.hook.eval import AquaTrajectoryEvalHook
from robonav.aqua.loss import AquaLoss
from robonav.aqua.model.aqua import AquaNet
from robonav.aqua.model.trajectory_head import TrajectoryHead
from robonav.aqua.runner.loop import AquaSequenceValLoop
from robonav.common.model.data_preprocessor import FrameBatchMerger

TERMS = ("traj_xy", "traj_yaw", "traj_unit", "traj_vel", "kin_pos", "kin_yaw", "depth")


def _seven(xy, yaw, vel, omega, q_scale=1.0):
    x, y = xy
    vx, vy = vel
    return torch.tensor(
        [x, y, math.sin(yaw) * q_scale, math.cos(yaw) * q_scale, vx, vy, omega],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Shared trajectory-metric calculation
# ---------------------------------------------------------------------------


def test_shared_calculator_identical_through_hook_and_metric_adapters():
    torch.manual_seed(0)
    pred7 = torch.stack(
        [
            _seven((0.5, 0.2), 3.1, (1.0, -0.3), 0.4, q_scale=0.8),
            _seven((1.0, 0.0), 0.2, (0.0, 2.0), -0.6),
        ]
    ).unsqueeze(0)
    gt7 = torch.stack(
        [
            _seven((0.0, 0.1), -3.1, (1.2, 0.0), 0.5),
            _seven((0.3, 0.0), 0.1, (0.0, 1.0), -0.5),
        ]
    ).unsqueeze(0)
    twist = torch.tensor([0.2, -0.1, 0.3])

    hook_metrics = AquaTrajectoryEvalHook(delta_t=0.1)._frame_metrics(pred7, gt7, twist)

    metric = AquaTrajectoryMetric(delta_t=0.1)
    metric.process(
        data_batch=[dict(future_trajectory=gt7[0], twist=twist)],
        data_samples=[dict(pred_trajectory=pred7[0])],
    )
    val_metrics = metric.compute_metrics(metric.results)

    assert set(hook_metrics) == set(val_metrics)
    for name in hook_metrics:
        assert hook_metrics[name] == pytest.approx(val_metrics[name])

    direct = frame_trajectory_metrics(pred7, gt7, twist.unsqueeze(0), 0.1)
    for name in hook_metrics:
        assert direct[name] == pytest.approx(hook_metrics[name])


def test_metric_averages_over_samples():
    metric = AquaTrajectoryMetric(delta_t=0.1)
    perfect = torch.zeros(1, 3, 7)
    perfect[..., 3] = 1.0
    off = perfect.clone()
    off[..., 0] = 2.0
    for pred in (perfect, off):
        metric.process(
            data_batch=[dict(future_trajectory=perfect[0], twist=None)],
            data_samples=[dict(pred_trajectory=pred[0])],
        )
    metrics = metric.compute_metrics(metric.results)
    assert metrics["ADE_m"] == pytest.approx(1.0)  # mean of 0.0 and 2.0


# ---------------------------------------------------------------------------
# mode="predict" output contract
# ---------------------------------------------------------------------------


class _StubBackbone(torch.nn.Module):
    def forward(self, camera_images, position_embedding):
        feat = camera_images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        return feat, feat, feat, feat


class _StubFeatureModulation(torch.nn.Module):
    def forward(self, f4, f3, twist, goal):
        return f3


class _StubTemporalFuser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, f3g, twist, delta_poses, goal):
        return torch.ones(twist.shape[0], 8), torch.zeros(twist.shape[0], 4)


class _StubDepthHead(torch.nn.Module):
    def forward(self, f4, f3, f2, f1):
        return (f4, f3)


def _loss():
    return AquaLoss(
        loss_weights={name: 1.0 for name in TERMS},
        depth_loss=dict(
            type="robonav.MultiScaleDepthLoss", max_depth=5, log_offset=0.1
        ),
        delta_t=0.1,
    )


def _predict_stub(num_steps):
    model = AquaNet.__new__(AquaNet)
    torch.nn.Module.__init__(model)
    model.backbone = _StubBackbone()
    model.feature_modulation = _StubFeatureModulation()
    model.temporal_fuser = _StubTemporalFuser()
    model.depth_head = _StubDepthHead()
    model.loss_module = _loss()
    model.trajectory_head = TrajectoryHead(
        feat_chans=8,
        hidden_chans=4,
        intermediate_chans=4,
        out_chans=7,
        num_trajectory_steps=num_steps,
    )
    return model


def _predict_inputs(batch, num_steps):
    return dict(
        # positive pixels keep the stub depth prediction log-valid
        camera_images=[torch.rand(1, 3, 8, 8) for _ in range(batch)],
        position_embedding=[torch.randn(1, 6, 4, 4) for _ in range(batch)],
        goal=[torch.randn(6) for _ in range(batch)],
        twist=[torch.randn(3) for _ in range(batch)],
        delta_poses=[torch.zeros(3) for _ in range(batch)],
        camera_depths=[torch.rand(1, 1, 2, 2) for _ in range(batch)],
        camera_depth_valid_masks=[
            torch.ones(1, 1, 2, 2, dtype=torch.bool) for _ in range(batch)
        ],
        occupancy=[torch.zeros(1) for _ in range(batch)],
        clearance=[torch.zeros(1) for _ in range(batch)],
        traversability=[torch.zeros(1) for _ in range(batch)],
        future_trajectory=[torch.randn(num_steps, 7) for _ in range(batch)],
    )


def test_predict_output_contract():
    num_steps = 4
    model = _predict_stub(num_steps)
    model.eval()
    for batch in (1, 3):
        inputs = _predict_inputs(batch, num_steps)
        losses = model(**inputs, mode="loss")
        outputs = model(**inputs, mode="predict")

        assert len(outputs) == batch + 1
        loss_element = outputs[-1]
        assert isinstance(loss_element, BaseDataElement)
        assert loss_element.keys() == ["loss"]
        assert set(loss_element.loss) == {f"loss_{name}" for name in TERMS}
        for name, value in loss_element.loss.items():
            assert not value.requires_grad
            assert torch.isfinite(value)
            torch.testing.assert_close(value, losses[name])
        # raw metrics and diagnostics must stay out of the loss element
        assert not any(("_raw" in k) or ("ADE" in k) for k in loss_element.loss)

        for sample in outputs[:-1]:
            assert isinstance(sample, BaseDataElement)
            assert sample.pred_trajectory.shape == (num_steps, 7)
            assert sample.pred_trajectory.device.type == "cpu"
            assert not sample.pred_trajectory.requires_grad

        # mode="tensor" behavior is unchanged
        trajectory = model(**inputs, mode="tensor")
        assert trajectory.shape == (batch, num_steps, 7)
        torch.testing.assert_close(outputs[0].pred_trajectory, trajectory[0].cpu())


def test_frame_forward_resets_only_on_stream_start():
    """Frame-oriented AquaNet owns the reset: it happens immediately before
    a stream_start frame and never mid-stream."""
    model = _predict_stub(4)
    model.eval()

    sentinel = object()
    model.temporal_fuser.state = sentinel
    model(**_predict_inputs(1, 4), mode="predict")
    assert model.temporal_fuser.state is sentinel  # continuation: no reset

    model(**_predict_inputs(1, 4), mode="predict", stream_start=True)
    assert model.temporal_fuser.state is None  # boundary: AquaNet reset


# ---------------------------------------------------------------------------
# AquaSequenceValLoop
# ---------------------------------------------------------------------------


class FakeValModel:
    """val_step returns one prediction per sample plus the trailing loss
    element. Like BaseModel.val_step it runs the real FrameBatchMerger, and
    like AquaNet it owns its reset: it performs it when the merged frame
    batch carries the batch-level stream_start boundary."""

    def __init__(self):
        self.data_preprocessor = FrameBatchMerger(device="cpu")
        self.reset_count = 0
        self.frames = []

    def eval(self):
        pass

    def reset(self):
        self.reset_count += 1

    def val_step(self, frame_batch):
        merged = self.data_preprocessor(frame_batch, training=False)
        if merged["stream_start"]:
            self.reset()
        self.frames.append(frame_batch)
        predictions = []
        for _ in frame_batch:
            prediction = BaseDataElement()
            prediction.pred_trajectory = torch.zeros(2, 7)
            predictions.append(prediction)
        loss_element = BaseDataElement()
        loss_element.loss = {
            "loss_a": torch.tensor(1.0),
            "loss_b": torch.tensor(2.0),
        }
        return predictions + [loss_element]


class RecordingEvaluator(Evaluator):
    def __init__(self):
        super().__init__(metrics=[])
        self.samples = []
        self.batches = []
        self.size = None

    def process(self, data_samples, data_batch):
        self.samples.extend(data_samples)
        self.batches.append(data_batch)

    def evaluate(self, size):
        self.size = size
        return {"trajectory/ADE_m": 0.5}


class FakeDataset:
    metainfo = {}

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length


class FakeSequenceDataLoader:
    def __init__(self, sequence_batches, sequence_length):
        self.sequence_batches = sequence_batches
        self.dataset = FakeDataset(sequence_length)

    def __len__(self):
        return len(self.sequence_batches)

    def __iter__(self):
        return iter(self.sequence_batches)


def _occurrence(stream_start=False):
    """Minimal Prefusion IndexInfo occurrence honoring the sampler contract."""
    occurrence = IndexInfo("scene", "frame")
    occurrence.stream_start = stream_start
    return occurrence


def _frame(batch, stream_start=False):
    return [
        dict(
            index_info=_occurrence(stream_start),
            future_trajectory=torch.zeros(2, 7),
            twist=torch.zeros(3),
        )
        for _ in range(batch)
    ]


def _make_loop(model, evaluator, sequence_batches, sequence_length):
    dataloader = FakeSequenceDataLoader(sequence_batches, sequence_length)
    runner = types.SimpleNamespace(
        model=model,
        visualizer=types.SimpleNamespace(),
        hooks=[],
    )
    runner.call_hook = lambda name, **kwargs: runner.hooks.append(
        (name, kwargs.get("batch_idx"))
    )
    return AquaSequenceValLoop(runner, dataloader, evaluator), runner


def test_val_loop_resets_processes_and_reports_global_count():
    model = FakeValModel()
    evaluator = RecordingEvaluator()
    # Arbitrary per-sequence batch sizes (2 and 3: the 3 includes an
    # accepted duplicated sequence slot), 3 frames each. Only the first
    # frame of each validation clip carries stream_start.
    sequence_batches = [
        [_frame(2, stream_start=(t == 0)) for t in range(3)],
        [_frame(3, stream_start=(t == 0)) for t in range(3)],
    ]
    loop, runner = _make_loop(model, evaluator, sequence_batches, sequence_length=3)

    metrics = loop.run()

    # exactly one model reset per validation clip, no mid-clip reset; the
    # loop itself never resets the model
    assert model.reset_count == 2
    assert len(model.frames) == 6  # every frame of every sequence
    assert loop._num_samples == 15  # 2*3 + 3*3 accepted samples
    assert len(evaluator.samples) == 15
    assert evaluator.size == 15  # actual processed count, not nominal
    assert len(evaluator.batches) == 6  # one process call per frame

    assert metrics["loss_a"] == pytest.approx(1.0)
    assert metrics["loss_b"] == pytest.approx(2.0)
    assert metrics["val_loss"] == pytest.approx(3.0)
    assert metrics["trajectory/ADE_m"] == pytest.approx(0.5)

    # iteration hooks fire once per frame with flattened indices, while
    # the temporal reset stays once per outer sequence batch
    iter_hooks = [(n, i) for n, i in runner.hooks if n.endswith("_val_iter")]
    assert [n for n, _ in iter_hooks] == [
        "before_val_iter",
        "after_val_iter",
    ] * 6
    assert [i for _, i in iter_hooks] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert [n for n, _ in runner.hooks][:2] == ["before_val", "before_val_epoch"]
    assert [n for n, _ in runner.hooks][-2:] == ["after_val_epoch", "after_val"]

    # accumulators stay device tensors until the single final reduce
    assert isinstance(loop._loss_sums["loss_a"], torch.Tensor)
    assert loop._loss_sums["loss_a"].device.type == "cpu"  # fixture loss device


# ---------------------------------------------------------------------------
# Distributed aggregation (2-process CPU gloo)
# ---------------------------------------------------------------------------


def _dist_worker(rank, world_size, init_file, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    loop = AquaSequenceValLoop.__new__(AquaSequenceValLoop)
    # rank 0: sums (a=10, b=1) over 5 samples; rank 1: (a=20, b=2) over 10.
    # Accumulators are device tensors, as left by run_iter.
    loop._loss_sums = {
        "loss_a": torch.tensor(float(10 * (rank + 1))),
        "loss_b": torch.tensor(float(rank + 1)),
    }
    loop._num_samples = 5 * (rank + 1)
    metrics, count = loop._reduce_stats()
    queue.put((metrics, count))
    dist.destroy_process_group()


def test_reduce_stats_two_ranks_gloo(tmp_path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    init_file = str(tmp_path / "dist_init")
    procs = [
        ctx.Process(target=_dist_worker, args=(rank, 2, init_file, queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    assert all(proc.exitcode == 0 for proc in procs)

    results = [queue.get() for _ in range(2)]
    for metrics, count in results:
        assert count == 15
        assert metrics["loss_a"] == pytest.approx(30 / 15)
        assert metrics["loss_b"] == pytest.approx(3 / 15)
        assert metrics["val_loss"] == pytest.approx(30 / 15 + 3 / 15)


# ---------------------------------------------------------------------------
# Registry / config wiring
# ---------------------------------------------------------------------------


def test_registry_and_config_wiring():
    import robonav  # noqa: F401
    from prefusion.registry import LOOPS as PREFUSION_LOOPS
    from prefusion.registry import METRICS as PREFUSION_METRICS
    from robonav.registry import LOOPS, METRICS

    assert LOOPS.get("AquaSequenceValLoop") is AquaSequenceValLoop
    assert PREFUSION_LOOPS.get("robonav.AquaSequenceValLoop") is AquaSequenceValLoop
    assert METRICS.get("AquaTrajectoryMetric") is AquaTrajectoryMetric
    assert PREFUSION_METRICS.get("robonav.AquaTrajectoryMetric") is AquaTrajectoryMetric
    assert AquaTrajectoryMetric.default_prefix == "trajectory"

    cfg = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "robonav",
            "aqua",
            "configs",
            "kinogoal_dla_resnet18_overfit.py",
        )
    )
    assert cfg.val_cfg["type"] == "robonav.AquaSequenceValLoop"
    assert cfg.val_evaluator["type"] == "robonav.AquaTrajectoryMetric"
    assert cfg.val_evaluator["delta_t"] == 0.1
    assert cfg.val_evaluator["prefix"] == "trajectory"
    sampler = cfg.val_dataloader["sampler"]
    assert sampler["type"] == "DefaultSampler"
    assert sampler["shuffle"] is False
    assert sampler["round_up"] is True
    assert cfg.train_cfg["val_interval"] == 50
    checkpoint = cfg.default_hooks["checkpoint"]
    assert checkpoint["save_best"] == "trajectory/ADE_m"
    assert checkpoint["rule"] == "less"
    assert cfg.val_dataset["batch_size"] == 2
    # test hook/setup untouched
    assert cfg.test_cfg["type"] == "SequenceBatchInferLoop"
    assert cfg.custom_hooks[0]["type"] == "robonav.AquaTrajectoryEvalHook"
    assert cfg.test_dataset["batch_size"] == 1
