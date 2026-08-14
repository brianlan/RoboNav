import importlib.util
import zipfile
from pathlib import Path

import numpy as np

_CONVERTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "dataset_converters"
    / "convert_sage3d_to_prefusion.py"
)


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_sage3d_to_prefusion", _CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(model, coefficients=None):
    profile = {
        "resolution": [64, 32],
        "model": model,
        "focal_length_pixels": [10.0, 10.0],
        "extrinsic": {
            "translation_body_m": [0.1, 0.2, 0.3],
            "rotation_rpy_deg": [1.0, 2.0, 3.0],
        },
    }
    if coefficients is not None:
        profile["fisheye_coefficients"] = coefficients
    return profile


def test_calibration_arrays_are_float32():
    converter = _converter()
    for model, coefficients in (
        ("pinhole", None),
        ("opencv_fisheye", [0.1, -0.01, 0.002, 0.001]),
    ):
        calibration = converter._calibration(_profile(model, coefficients))
        assert calibration["intrinsic"].dtype == np.float32
        rotation, translation = calibration["extrinsic"]
        assert rotation.dtype == np.float32
        assert translation.dtype == np.float32
        if coefficients is not None:
            assert calibration["intrinsic"].shape == (8,)
            assert np.allclose(calibration["intrinsic"][4:], coefficients)
        else:
            assert calibration["intrinsic"].shape == (4,)


def test_ego_pose_arrays_are_float32():
    converter = _converter()
    trajectory = {
        "pose_world": np.array([[1.0, 2.0, 0.5]]),
        "velocity_world_mps": np.array([[3.0, 4.0]]),
        "yaw_rate_radps": np.array([0.1]),
    }
    ego_pose = converter._ego_pose(trajectory, 0)
    assert set(ego_pose) == {
        "rotation",
        "translation",
        "linear_velocity",
        "angular_velocity",
    }
    for name, value in ego_pose.items():
        assert value.dtype == np.float32, name


def test_goal_is_relative_to_current_body_frame():
    converter = _converter()
    trajectory = {
        "pose_world": np.array([[1.0, 2.0, 0.3], [3.0, 5.0, 0.9]]),
        "velocity_world_mps": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "yaw_rate_radps": np.array([0.1, 0.4]),
    }
    ego = converter._ego_pose(trajectory, 0)
    terminal = converter._ego_pose(trajectory, 1)
    goal = converter._goal(ego, terminal)
    assert set(goal) == {
        "rotation",
        "translation",
        "linear_velocity",
        "angular_velocity",
    }
    for name, value in goal.items():
        assert value.dtype == np.float32, name
    assert goal["rotation"].shape == (3, 3)
    assert goal["translation"].shape == (3,)
    assert goal["linear_velocity"].shape == (3,)
    assert goal["angular_velocity"].shape == (3,)
    assert np.allclose(
        ego["rotation"] @ goal["translation"],
        terminal["translation"] - ego["translation"],
        atol=1e-6,
    )
    assert np.allclose(ego["rotation"] @ goal["rotation"], terminal["rotation"], atol=1e-6)
    assert np.allclose(
        ego["rotation"] @ goal["linear_velocity"], [3.0, 4.0, 0.0], atol=1e-6
    )
    assert np.allclose(
        ego["rotation"] @ goal["angular_velocity"], [0.0, 0.0, 0.4], atol=1e-6
    )
    terminal_goal = converter._goal(terminal, terminal)
    assert np.allclose(terminal_goal["translation"], 0.0, atol=1e-6)
    assert np.allclose(terminal_goal["rotation"], np.eye(3), atol=1e-6)
    assert np.allclose(
        terminal_goal["linear_velocity"], terminal["linear_velocity"], atol=1e-6
    )
    assert np.allclose(
        terminal_goal["angular_velocity"], terminal["angular_velocity"], atol=1e-6
    )


def test_ego_pose_visualization_geometry(tmp_path):
    converter = _converter()
    trajectory = {
        "time_s": np.array([0.0, 0.1, 0.2]),
        "pose_world": np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.3, 0.7], [1.0, 0.5, -0.4]]
        ),
        "velocity_world_mps": np.array([[0.3, 0.4], [0.5, -0.2], [0.1, 0.6]]),
        "yaw_rate_radps": np.array([0.3, -0.4, 0.2]),
    }
    frame_ids = ["1000", "1001", "1002"]
    fig = converter._ego_pose_visualization(tmp_path, trajectory, frame_ids)
    assert len(fig.frames) == 3
    assert [f.name for f in fig.frames] == frame_ids
    assert tuple(fig.frames[1].traces) == (1, 2, 3, 4, 5)

    steps = fig.layout.sliders[0].steps
    assert [s.args[0] == (fid,) for s, fid in zip(steps, frame_ids)] == [True] * 3
    play_duration = fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
    assert play_duration == 100

    i = 1
    t = trajectory["pose_world"][i]
    yaw = t[2]
    c, s = np.cos(yaw), np.sin(yaw)
    axis, vscale = 0.2, 1.0
    v_body = converter._ego_pose(trajectory, i)["linear_velocity"][:2]
    v_world = np.array([[c, -s], [s, c]]) @ v_body
    ang = yaw + trajectory["yaw_rate_radps"][i]

    data = fig.frames[i].data
    assert np.allclose(data[0].x, [t[0]]) and np.allclose(data[0].y, [t[1]])
    assert np.allclose(data[1].x, [t[0], t[0] + axis * c])
    assert np.allclose(data[1].y, [t[1], t[1] + axis * s])
    assert np.allclose(data[2].x, [t[0], t[0] - axis * s])
    assert np.allclose(data[2].y, [t[1], t[1] + axis * c])
    assert np.allclose(data[3].x, [t[0], t[0] + vscale * v_world[0]])
    assert np.allclose(data[3].y, [t[1], t[1] + vscale * v_world[1]])
    assert np.allclose(data[4].x, [t[0], t[0] + axis * np.cos(ang)])
    assert np.allclose(data[4].y, [t[1], t[1] + axis * np.sin(ang)])

    half = (fig.layout.xaxis.range[1] - fig.layout.xaxis.range[0]) / 2
    center = np.array(
        [
            np.mean(fig.layout.xaxis.range),
            np.mean(fig.layout.yaxis.range),
        ]
    )
    max_extent = vscale * np.max(np.linalg.norm(trajectory["velocity_world_mps"], axis=1))
    far = np.max(np.abs(trajectory["pose_world"][:, :2] - center))
    assert far + max_extent <= half + 1e-6

    assert not (tmp_path / "ego_pose_visualization.html").exists()
    zip_path = tmp_path / "ego_pose_visualization.zip"
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["ego_pose_visualization.html"]
        html = archive.read("ego_pose_visualization.html").decode("utf-8")
        info = archive.getinfo("ego_pose_visualization.html")
    assert "<script src=" not in html
    for fid in frame_ids:
        assert fid in html
    assert info.compress_size < info.file_size
