"""Merge per-scene info pickles under a scene root into a single pickle."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--info-name", default="info.pkl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_arguments(argv)
    scene_root = args.scene_root
    if not scene_root.is_dir():
        logger.error("scene root is not a directory: {}", scene_root)
        return 2

    scene_dirs = _select_scene_dirs(scene_root, args.scene_ids, args.info_name)
    if not scene_dirs:
        return 2

    try:
        infos = _load_infos(scene_dirs, args.info_name)
        merged = _merge(infos)
        _write(merged, args.dest)
    except Exception as error:
        logger.error("merge failed: {}", error)
        return 2

    frame_count = sum(len(info["frame_info"]) for info in merged.values())
    logger.info(
        "merged {} scenes, {} frames into {} in {:.1f}s",
        len(merged),
        frame_count,
        args.dest,
        time.perf_counter() - started,
    )
    return 0


def _select_scene_dirs(scene_root: Path, scene_ids: list[str] | None, info_name: str) -> list[Path]:
    if scene_ids is not None:
        missing = [
            scene_id
            for scene_id in scene_ids
            if not (scene_root / scene_id / info_name).is_file()
        ]
        if missing:
            logger.error("missing scenes or {} files: {}", info_name, missing)
            return []
        return [scene_root / scene_id for scene_id in scene_ids]
    scene_dirs = [d for d in sorted(scene_root.iterdir()) if (d / info_name).is_file()]
    ignored = sum(1 for d in scene_root.iterdir() if not (d / info_name).is_file())
    if ignored:
        logger.warning("ignored {} entries without {}", ignored, info_name)
    return scene_dirs


def _load_infos(scene_dirs: list[Path], info_name: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        return list(
            executor.map(lambda d: _load_info(d / info_name), scene_dirs)
        )


def _load_info(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _merge(infos: list[dict]) -> dict:
    merged: dict = {}
    duplicates = set()
    for info in infos:
        for key, value in info.items():
            if key in merged:
                duplicates.add(key)
            else:
                merged[key] = value
    if duplicates:
        raise ValueError(f"duplicate top-level keys across scenes: {sorted(duplicates)}")
    return dict(sorted(merged.items()))


def _write(merged: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(merged, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(dest)


if __name__ == "__main__":
    sys.exit(main())
