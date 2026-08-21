import importlib.util
import pickle
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "dataset_converters"
    / "merge_scene_infos.py"
)


def _script():
    spec = importlib.util.spec_from_file_location("merge_scene_infos", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_info(directory: Path, scene_id: str, info_name: str = "info.pkl") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    info = {scene_id: {"scene_info": {}, "frame_info": {f"{scene_id}/frame_info_pkl/1.pkl"}}}
    with (directory / info_name).open("wb") as stream:
        pickle.dump(info, stream)


def test_merges_scenes_in_sorted_order(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    for scene_id in ("b", "a"):
        _write_info(root / scene_id, scene_id)

    assert script.main(["--scene-root", str(root), "--dest", str(tmp_path / "out.pkl")]) == 0
    with (tmp_path / "out.pkl").open("rb") as stream:
        merged = pickle.load(stream)
    assert list(merged) == ["a", "b"]


def test_values_are_merged_verbatim(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    original = {"s1": {"scene_info": {}, "frame_info": {"1": "s1/frame_info_pkl/1.pkl"}}}
    (root / "s1").mkdir(parents=True)
    with (root / "s1" / "info.pkl").open("wb") as stream:
        pickle.dump(original, stream)

    assert script.main(["--scene-root", str(root), "--dest", str(tmp_path / "out.pkl")]) == 0
    with (tmp_path / "out.pkl").open("rb") as stream:
        merged = pickle.load(stream)
    assert merged == original


def test_explicit_scene_ids(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    for scene_id in ("s1", "s2"):
        _write_info(root / scene_id, scene_id)

    dest = tmp_path / "out.pkl"
    assert script.main(["--scene-root", str(root), "--scene-ids", "s2", "--dest", str(dest)]) == 0
    with dest.open("rb") as stream:
        merged = pickle.load(stream)
    assert list(merged) == ["s2"]


def test_missing_scene_id_fails(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    _write_info(root / "s1", "s1")

    assert script.main(["--scene-root", str(root), "--scene-ids", "s1", "gone", "--dest",
                        str(tmp_path / "out.pkl")]) == 2


def test_enumeration_ignores_directories_without_info(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    _write_info(root / "s1", "s1")
    (root / "not-a-scene").mkdir()

    dest = tmp_path / "out.pkl"
    assert script.main(["--scene-root", str(root), "--dest", str(dest)]) == 0
    with dest.open("rb") as stream:
        merged = pickle.load(stream)
    assert list(merged) == ["s1"]


def test_duplicate_keys_fail(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    for scene_id in ("dir1", "dir2"):
        _write_info(root / scene_id, "same-key")

    assert script.main(["--scene-root", str(root), "--dest", str(tmp_path / "out.pkl")]) == 2


def test_custom_info_name(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    _write_info(root / "s1", "s1", info_name="index.pkl")

    dest = tmp_path / "out.pkl"
    assert script.main(["--scene-root", str(root), "--dest", str(dest),
                        "--info-name", "index.pkl"]) == 0
    with dest.open("rb") as stream:
        merged = pickle.load(stream)
    assert list(merged) == ["s1"]


def test_overwrites_existing_dest(tmp_path):
    script = _script()
    root = tmp_path / "scenes"
    _write_info(root / "s1", "s1")
    dest = tmp_path / "out.pkl"
    dest.write_bytes(b"stale")

    assert script.main(["--scene-root", str(root), "--dest", str(dest)]) == 0
    with dest.open("rb") as stream:
        merged = pickle.load(stream)
    assert list(merged) == ["s1"]
