# AquaNet sequence BPTT and recurrent-state lifecycle design

## Status

Implemented follow-up to RoboNav commit `4cc9a65` (`temporary fix for training using MultiGPU`). The explicit recurrent-state boundary contract and the supporting Prefusion changes described below are implemented in both repositories. Full-sequence training remains one root DDP forward per BPTT sequence.

## Why this document exists

The original issue appeared only in multi-GPU training. Single-GPU training accepted a temporal sequence because `AquaNet.train_step()` understood the nested sequence, while MMEngine's distributed wrapper used its own `train_step()` and sent the same nested data through a frame-only preprocessor. That exposed assumptions such as `data[0]` and failed before the intended BPTT workflow could run.

The investigation expanded into three related questions:

1. How should one complete sequence pass through DDP while retaining a full BPTT graph?
2. Which Module should own the SRU reset lifecycle in train, validation, and test?
3. How should Prefusion represent a recurrent-state boundary without overloading mutable `q_prev` and `s_prev` links?

## Canonical terms

- **BPTT sequence**: an independent finite sequence used as one training unit, with one forward through the root DDP wrapper, one backward, and one optimizer update.
- **Validation clip**: an independent finite sequence evaluated frame by frame without gradients. Hidden state continues inside the clip and resets at its start.
- **Scene stream**: an ordered stream of test frames whose hidden state continues across dataset items and resets only when a new scene starts.
- **Frame batch**: samples from multiple aligned streams at one temporal offset.
- **Sequence batch**: multiple equal-length sequences aligned by temporal offset.
- **`stream_start`**: a batch-aligned model input declaring that the current frame starts a new recurrent-state lifetime.
- **`q_prev` / `q_next`**: adjacency of a frame occurrence inside a sampled sequence.
- **`s_prev` / `s_next`**: adjacency of frames inside a scene.

The dataset's singleton test sequence is a packaging unit, not necessarily a recurrent-state boundary. The scene stream is the relevant state lifetime during test.

## Current train, validation, and test behavior

### Training

`StreamingSequenceBatchDataset` and `AlignedTimestepBatchSampler` produce one aligned frame microbatch at a time. `StreamingSequenceBPTTTrainLoop` collects `T` consecutive microbatches into an explicit `{"sequence": [...]}` raw envelope and calls `train_step()` once. `FrameBatchMerger` dispatches on that envelope field (never on the `training` flag), merges each inner frame batch, and `AquaNet.forward(sequence=...)` processes all frames in order.

The sequence takes exactly one root DDP forward. Inside that forward, every frame contributes to one connected recurrent graph; losses are averaged over time and the default MMEngine training path performs one backward and one optimizer update. `find_unused_parameters=True` sees the union of the whole sequence graph rather than only the last frame.

`AquaNet._forward_sequence()` resets before and after the finite sequence and ignores the per-frame `stream_start` markers (without mutating the processed frame dicts), so the first frame is not reset twice.

The `4cc9a65` baseline was verified at the time with the targeted sequence/DDP tests, the RoboNav suite (then `130 passed, 1 skipped`; the skip requires optional `onnx`), and a real two-GPU NCCL recurrent smoke test that observed one root DDP forward per sequence, synchronized parameters, and correct handling of a parameter used only by an early frame with `find_unused_parameters=True`. This revision did not re-run a real NCCL smoke test (see [Verification](#verification-current-revision)).

### Validation

`SequenceBatchDataset` returns a complete sequence batch. `AquaSequenceValLoop` calls `val_step()` once per frame batch in order and never resets the model. Each frame batch flows through `FrameBatchMerger`, which adapts the sampler occurrences to a batch-level `stream_start`; `AquaNet` resets itself immediately before processing the first frame of each validation clip. `TemporalFuser.state` persists across the remaining calls because nothing resets mid-clip.

The orchestration is sequence-aware—`AquaSequenceValLoop` iterates a complete validation clip in order—while each `AquaNet` call is frame-oriented. The `stream_start` boundary lets model-owned state persist across those calls within a clip and reset correctly at its start.

### Test

`SequentialSceneFrameSequenceSampler` returns singleton dataset sequences in scene/frame order, establishes `s_prev` / `s_next` across the scene, and marks `stream_start` on the first frame of each scene (despite `sequence_length=1`). `SequenceBatchInferLoop` invokes `test_step()` once for each singleton frame; `FrameBatchMerger` adapts each occurrence to a batch-level `stream_start`, and `AquaNet` resets at scene starts, so state continues across singleton dataset items in the same scene. `AquaTrajectoryEvalHook` no longer resets or tracks scene boundaries; it accumulates metrics only.

This is correct for the configured single-rank ordered test path. A normal distributed frame-level `DefaultSampler` would shard consecutive scene frames across ranks and cannot preserve full-scene recurrent state; multi-GPU test requires scene-level sharding and is outside the current change.

## Why the sequence-aware DDP design was selected

### Selected design

Make `AquaNet.forward()` sequence-aware and let the existing single-GPU `BaseModel.train_step()` and multi-GPU `MMDistributedDataParallel.train_step()` own preprocessing, loss parsing, AMP, gradient accumulation, clipping, backward, synchronization, and optimizer stepping.

The root DDP wrapper is crossed exactly once per sequence.

### Rejected: one DDP forward per frame, one backward after all frames

This performs DDP pre/post-forward bookkeeping `T` times and interacts poorly with `find_unused_parameters=True`. If a parameter is used in an earlier frame but not the last frame, repeated reducer preparation can report that a variable was marked ready twice. It also couples the recurrent implementation to DDP reducer details and adds avoidable overhead.

### Rejected: call `ddp.module.train_step()` directly

This bypasses the root DDP forward and therefore bypasses reducer preparation. Backward gradients are not synchronized across ranks.

### Rejected: nested DDP submodules

This complicates reset access and checkpoint key handling. Default checkpoint save/load does not provide a clean nested-wrapper round trip without extra state-dict adapters.

### Rejected: manual gradient all-reduce or per-frame optimizer steps

Manual synchronization duplicates framework behavior and must reimplement AMP, clipping, accumulation, and unused-parameter semantics. Per-frame optimizer steps break full-sequence BPTT.

## Streaming dataset versus SequenceBatchDataset for training

The current training path no longer performs streamed computation: the loop collects the complete sequence before the model runs, the preprocessor moves the complete sequence to the device, and BPTT retains all temporal activations.

Streaming still affects the host-side data pipeline:

- a worker task loads one aligned frame batch rather than an entire sequence batch;
- the prefetch queue holds frame microbatches rather than `T` complete sequence batches;
- heavy image/depth transforms have smaller per-worker peak memory and finer scheduling granularity.

With batch size 4, sequence length 20, eight workers, and high-resolution camera/depth inputs, this host-memory difference may be material. `SequenceBatchDataset` would simplify the Interface and make train and validation data shapes more consistent, but each prefetched worker item would contain `B * T` frames.

No dataset migration is part of the state-boundary change. A later migration should be decided by measuring host RSS, dataloader wait time, and sequences per second. Prefusion's existing `SequenceBatchTrainLoop` cannot be used for full BPTT because it calls `train_step()` once per frame; a sequence-level BPTT loop would still be required.

## Prefusion q-link defect discovered during design

`TrainIndexSequenceSampler` and `ValIndexSequenceSampler` first build sequences from shared scene `IndexInfo` objects, then call `establish_sequence_linkings()` on every sequence. Generated sequences may overlap, especially at padded scene tails or with different frame intervals.

For example:

```text
sequence A = [0, 1, 2, 3]
sequence B = [2, 3, 4, 5]
```

Frames 2 and 3 are the same Python objects in both sequences. Establishing `q_prev` / `q_next` for sequence B overwrites the links previously established for sequence A. Depending on construction order, a sequence start can acquire a non-null `q_prev`, or an interior frame can acquire `q_prev=None`.

`establish_linkings()` clones duplicate occurrences only within one list and never isolated occurrences shared across different sequence lists.

The sequence list itself is still consumed in the correct order, and both `AlignedTimestepBatchSampler` and `SequenceBatchDataset` correctly align equal temporal offsets across batch positions. The defect is in the mutable link metadata, not in batch ordering.

This defect affects more than reset inference: loaders that traverse `IndexInfo.prev` for historical data can observe a link written by another sampled sequence.

Fixed in this revision: sampled sequences are linked through `establish_isolated_sequence_occurrences()` at the linking/sampling seam, which gives every sampled sequence its own independent occurrences (scene links preserved, stale q-links cleared) before q-links are established (see [Prefusion repository](#prefusion-repository)).

## Final target design

The design below is implemented in both repositories (see [Implemented cross-repository changes](#implemented-cross-repository-changes)).

### State-boundary ownership

All physical SRU resets should be executed by `AquaNet`:

- `forward(sequence=...)` represents an independent finite BPTT sequence and owns its start/end reset;
- frame-oriented forward accepts `stream_start` and resets immediately before processing a starting frame;
- validation and test orchestration no longer call `model.reset()` directly.

The orchestration/data pipeline declares the semantic boundary; the model hides how recurrent state is reset.

### Boundary source

Do not make `AquaNet` understand Prefusion's `IndexInfo`, `q_prev`, or `s_prev`. Prefusion should explicitly mark the state boundary, and `FrameBatchMerger` should adapt that Prefusion fact to AquaNet's `stream_start` Interface.

The boundary semantics differ by sampler:

- `TrainIndexSequenceSampler`: the first occurrence of each sampled sequence starts state;
- `ValIndexSequenceSampler`: the first occurrence of each validation clip starts state;
- `SequentialSceneFrameSequenceSampler`: the first frame of each scene starts state even though every dataset sequence has length one.

This is real behavior variation at the sampler Seam, not a speculative abstraction.

### IndexInfo contract

`index_info` is mandatory in the normal Prefusion data path. Missing metadata is unsupported; no fallback reset policy will be added. A missing required field should fail naturally at the Adapter rather than silently defaulting to reset or continuation.

`IndexInfo` declares `stream_start` as an explicit optional field (`__init__` and `from_str` parameter, default `None`). `None` means the boundary semantics have not been assigned yet; the sequence samplers replace it with a real Python bool. The field is occurrence metadata only: `IndexInfo.__eq__`, `__repr__`, `prev` / `next`, and `as_dict` intentionally do not include it.

A sequence-aware consumer requires its configured `ModelFeeder` to preserve `index_info`. `AquaModelFeeder` does; generic/stateless feeders may omit it. The dataset does not override or reattach feeder output, and no `_meta` duplication or recovery of `index_info` / `stream_start` exists. A feeder that drops `index_info` is therefore an invalid configuration for `FrameBatchMerger` and fails naturally on the missing key at the Adapter. The reviewer proposal to recover `index_info` at the dataset level was considered and rejected as out of scope for this consumer contract: it would add dataset-level bookkeeping and `_meta` duplication for a misconfiguration, contradicting the rule that the feeder owns the model-input contract.

`q_prev` and `s_prev` remain useful data-topology links, but recurrent-state lifetime must not rely on `prev = q_prev or s_prev`. A nullable `q_prev` cannot distinguish “no sequence context” from “first occurrence in a sequence,” especially if an occurrence carries both scene and sequence links.

## Implemented cross-repository changes

### Prefusion repository

Repository: `/home/rlan/projects/prefusion`

Branch: `fix/sequence-stream-start` (from clean `main` at `a57cc4b` (`v2.4.6`)).

1. Every sampled sequence owns independent `IndexInfo` occurrences before q-links are established. `establish_isolated_sequence_occurrences()` at the linking/sampling seam shallow-copies each occurrence (scene `s_prev` / `s_next` preserved, stale q-links cleared) and q-links the copies, so no sampled sequence can overwrite another's links; repeated padded occurrences each get their own object and cannot self-loop. The helper is not exported through `__all__`.
2. Explicit occurrence-level `stream_start` metadata is part of the sampler output contract. Train/validation mark sequence position zero; sequential-scene sampling marks only scene starts. The representation is an explicit optional `stream_start` field on `IndexInfo` (constructor and `from_str` parameter, default `None`), which the samplers replace with a real bool; `ClassBalancedSequenceSampler` inherits the semantics from its base `TrainIndexSequenceSampler`, and deep-copied oversampled duplicates preserve the markers.
3. The existing `index_info` requirement is unchanged. With the configured `AquaModelFeeder`, the metadata rides `index_info` through the streaming and sequence-batch paths; generic feeders remain responsible for their own model-input interface, and there is no dataset fallback.
4. Focused tests added:
   - overlapping sequences have independent, complete q-chains;
   - padded/repeated occurrences do not create self-loops;
   - scene links survive occurrence copying;
   - train and validation mark only sequence starts;
   - sequential-scene sampling marks only scene starts;
   - equal temporal offsets in one sequence batch expose the same boundary value;
   - class-balanced oversampling preserves `stream_start` semantics.
5. The Prefusion dataset/sampler test set and the feasible full suite have been run (see [Verification](#verification-current-revision)).

The exact representation is the minimal one: an explicit optional `stream_start` field on the sampled `IndexInfo` occurrence, because RoboNav's feeder already preserves `index_info`. No wrapper type was introduced.

### RoboNav repository

Repository: `/data/home/rlan/projects/RoboNav`

1. `FrameBatchMerger` no longer uses `training` as a data-shape discriminator. `StreamingSequenceBPTTTrainLoop` passes an explicit `{"sequence": [...]}` raw envelope; the merger dispatches on that field and merges each inner frame batch.
2. `FrameBatchMerger._merge_frame()` reads the Prefusion occurrence's explicit `stream_start`, requires every occurrence to carry a real bool (unassigned `None` is rejected, not coerced), verifies that all samples in a time-aligned frame batch agree (disagreement raises `ValueError` instead of silently corrupting recurrent state), and emits one batch-level boolean. A missing required field fails naturally at this Adapter seam.
3. AquaNet's frame Interface accepts `stream_start`; `_forward_frame()` resets immediately before processing a starting frame.
4. `forward(sequence=...)` stays self-contained: it resets at finite-sequence boundaries and ignores per-frame `stream_start` without mutating the processed frame dicts, so the first frame is not reset twice.
5. `AquaSequenceValLoop` no longer resets; the first frame's boundary signal drives the model-owned reset.
6. `AquaTrajectoryEvalHook` retains metric accumulation only; reset and scene-tracking responsibility is gone.
7. Tests and fixtures honor the mandatory `index_info` contract with real or minimal `IndexInfo` occurrences.

### Integration status

1. The Prefusion contract was implemented and tested first, on branch `fix/sequence-stream-start`.
2. RoboNav was implemented against the Prefusion working tree. Because the Python environment imports a regular site-packages Prefusion copy (not an editable install), cross-repository tests used `PYTHONPATH` to include both working trees; real RoboNav training/test runs require the Prefusion branch or a release installed.
3. A real two-GPU NCCL smoke test remains optional operational verification; it is not part of the implementation and not a prerequisite for committing.

## Feasibility and operational constraints

- RoboNav implements the changes on branch `fix/aquanet-sequence-stream-start` with open PR https://github.com/brianlan/RoboNav/pull/1.
- Prefusion implements the changes on branch `fix/sequence-stream-start` (from clean `main` at `a57cc4b` (`v2.4.6`)) with open PR https://github.com/Auto-AI-Ragtag/prefusion/pull/4.
- SSH authentication to `github.com` succeeds as `brianlan`; the `gh` CLI is installed and authenticated, so branch creation, commit, push, and PR creation are available.

## Verification (current revision)

All results below use `/data/home/rlan/envs/prefusion_py311/bin/python`.

- RoboNav full suite with both working trees on `PYTHONPATH`: `136 passed, 1 skipped` (the skip requires optional `onnx`).
- Prefusion affected IndexInfo/sampler/dataset focused set: `109 passed`.
- Feasible full Prefusion run: `399 passed, 11 failed` (known pre-existing failures unrelated to this change: 5 lidar loader tests and 6 metric AP tests), plus the same collection limitation in `tests/prefusion/utils/test_generate_visiblity.py` because the optional `mtv4d` module is not installed in the environment.
- Two-rank Gloo DDP coverage (`find_unused_parameters=False` and `True`) passes: one root DDP forward per sequence, gradients equivalent to a global-batch reference, synchronized parameters, and correct gradients for a parameter used only by the first frame.
- No real NCCL multi-GPU smoke test was rerun: no committed smoke-test script exists in the documented workflow, and real training was not started.

## Decision summary

The model should own recurrent-state mutation, while samplers own the meaning of temporal boundaries and the data Adapter translates between them. Full-sequence training remains one root DDP forward so DDP sees the complete temporal graph. Prefusion now isolates sampled sequence occurrences because mutable q-links on shared `IndexInfo` objects violated the intended sequence contract. An explicit `stream_start` removes the ambiguous `q_prev or s_prev` inference and lets train, validation, and scene-stream test use one small AquaNet Interface without conflating their different sequence meanings.
