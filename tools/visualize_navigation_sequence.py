"""Render post-transform NavigationMap2D sequences with body-frame labels."""

import argparse
import copy
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from mmengine.config import Config, DictAction
from prefusion.registry import DATASETS

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robonav  # noqa: F401


OCCUPANCY_COLORS = ("#9e9e9e", "#f5f5f5", "#d94841")


def front_up_raster(plane):
    """Return a [X, Y] body raster in the front-up display orientation."""
    return np.flip(np.asarray(plane), axis=1)


def map_point(point):
    x, y = np.asarray(point)[:2]
    return -float(y), float(x)


def map_vector(vector):
    return map_point(vector)


def map_extent(output_range):
    xmin, ymin, xmax, ymax = output_range
    return (-ymax, -ymin, xmin, xmax)


def decode_occupancy(occupancy):
    """Decode [unknown, free, occupied] one-hot channels to category ids."""
    channels = np.asarray(occupancy)
    return np.argmax(channels, axis=0)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _draw_frame(ax, point, rotation, scale=0.35):
    px, py = map_point(point)
    rotation = _as_numpy(rotation)
    for column, color in ((0, "#1565c0"), (1, "#ef6c00")):
        dx, dy = map_vector(rotation[:2, column])
        ax.arrow(px, py, scale * dx, scale * dy, color=color, width=0.012,
                 head_width=0.11, length_includes_head=True, zorder=4)


def _overlay(ax, frame):
    ax.arrow(0, 0, 0, 1, color="#1565c0", width=0.018, head_width=0.14,
             length_includes_head=True, zorder=5)
    ax.arrow(0, 0, -1, 0, color="#ef6c00", width=0.018, head_width=0.14,
             length_includes_head=True, zorder=5)

    transformables = frame["transformables"]
    for transformable in transformables.values():
        name = type(transformable).__name__
        if name == "FutureTrajectory":
            for point, rotation in zip(transformable.translation, transformable.rotation):
                _draw_frame(ax, point, rotation, scale=0.22)
            points = [map_point(point) for point in transformable.translation]
            if points:
                ax.plot(*zip(*points), color="#7b1fa2", linewidth=1.2, zorder=3)
        elif name == "Goal":
            point = _as_numpy(transformable.translation).reshape(-1)
            ax.scatter(*map_point(point), marker="*", s=90, color="#c62828", zorder=5)
            _draw_frame(ax, point, transformable.rotation, scale=0.38)


def render_frame(frame, output_dir, frame_number, frame_id, output_range, clearance_vmax):
    tensors = frame["occupancy"], frame["clearance"], frame["traversability"]
    arrays = [
        decode_occupancy(_as_numpy(tensors[0])),
        _as_numpy(tensors[1]).squeeze(0),
        _as_numpy(tensors[2]).squeeze(0),
    ]
    kinds = ("occupancy", "clearance", "traversability")
    colormaps = (ListedColormap(OCCUPANCY_COLORS), "viridis", "binary")
    limits = ((0, 2), (0, clearance_vmax), (0, 1))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(frame_id))
    for kind, array, cmap, (vmin, vmax) in zip(kinds, arrays, colormaps, limits):
        kind_dir = output_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(front_up_raster(array), origin="lower", extent=map_extent(output_range),
                  cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        _overlay(ax, frame)
        ax.set_aspect("equal")
        ax.set_xlim(map_extent(output_range)[:2])
        ax.set_ylim(map_extent(output_range)[2:])
        ax.set_xlabel("-body Y (m); body +Y points left")
        ax.set_ylabel("body +X forward (m)")
        fig.savefig(kind_dir / f"frame-{frame_number:04d}-{safe_id}-{kind}.png", dpi=120)
        plt.close(fig)


def _build_dataset(config_path, dataset_key, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    dataset_cfg = copy.deepcopy(cfg[dataset_key])
    feeder_cfg = copy.deepcopy(dataset_cfg.get("model_feeder"))
    if feeder_cfg is not None:
        feeder_cfg["debug"] = True
        dataset_cfg["model_feeder"] = feeder_cfg
    dataset = DATASETS.build(dataset_cfg)
    smith_cfg = cfg[dataset_key].transformables.navigation_map_2d.tensor_smith
    return dataset, tuple(smith_cfg.output_range)


def _sequence_frames(dataset, sequence_index, epoch):
    if type(dataset).__name__ == "StreamingSequenceBatchDataset":
        dataset.set_epoch(epoch)
        start = sequence_index * dataset.sequence_length
        return [dataset[start + offset] for offset in range(dataset.sequence_length)]

    batch_size = dataset.batch_size
    batch = dataset[sequence_index // batch_size]
    return [frame[sequence_index % batch_size] for frame in batch]


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("dataset_key", nargs="?", default="train_dataset")
    parser.add_argument("sequence_index", nargs="?", type=int, default=0)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("navigation_sequence"))
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    parsed = parser.parse_args(args)
    dataset, output_range = _build_dataset(parsed.config, parsed.dataset_key, parsed.cfg_options)
    frames = _sequence_frames(dataset, parsed.sequence_index, parsed.epoch)
    vmax = max(float(_as_numpy(frame["clearance"]).max()) for frame in frames)
    vmax = vmax if vmax > 0 else 1.0
    for number, frame in enumerate(frames):
        frame_id = getattr(frame["index_info"], "scene_frame_id", number)
        render_frame(frame, parsed.output_dir, number, frame_id, output_range, vmax)


if __name__ == "__main__":
    main()
