import importlib.util
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
