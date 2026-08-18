import importlib.util
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image

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


def _trajectory():
    return {
        "pose_world": np.array(
            [[0.0, 0.0, 0.3], [3.0, 5.0, 0.9], [1.0, -1.0, -0.4], [2.0, 2.0, 0.0]]
        ),
        "velocity_world_mps": np.array(
            [[1.0, 2.0], [3.0, 4.0], [0.5, -0.5], [1.0, 0.0]]
        ),
        "yaw_rate_radps": np.array([0.1, 0.4, -0.3, 0.2]),
    }


def test_future_trajectory_matches_goal_and_padding():
    converter = _converter()
    trajectory = _trajectory()

    def check(frame_index, num, source_indices):
        states = converter._future_trajectory(trajectory, frame_index, num)
        assert set(states) == {
            "rotation",
            "translation",
            "linear_velocity",
            "angular_velocity",
        }
        ego = converter._ego_pose(trajectory, frame_index)
        for name, value in states.items():
            assert value.dtype == np.float32, name
            expected_shape = (num, 3, 3) if name == "rotation" else (num, 3)
            assert value.shape == expected_shape, name
        for offset, source in enumerate(source_indices):
            expected = converter._goal(ego, converter._ego_pose(trajectory, source))
            for name, value in states.items():
                np.testing.assert_allclose(value[offset], expected[name], atol=1e-6)

    check(1, 4, [2, 3, 3, 3])
    check(3, 2, [3, 3])


def test_parse_arguments_num_future_trajectory_steps():
    converter = _converter()
    required = ["--input-scene-root", "a", "--output-scene-root", "b"]
    args = converter.parse_arguments(required)
    assert args.num_future_trajectory_steps == 20
    assert args.visualize_future_trajectory is False
    assert args.visualize_ego_pose is False
    assert (
        converter.parse_arguments(
            required + ["--num-future-trajectory-steps", "7"]
        ).num_future_trajectory_steps
        == 7
    )
    assert (
        converter.parse_arguments(
            required + ["--visualize-future-trajectory"]
        ).visualize_future_trajectory
        is True
    )
    assert (
        converter.parse_arguments(required + ["--visualize-ego-pose"]).visualize_ego_pose is True
    )
    both = converter.parse_arguments(
        required + ["--visualize-future-trajectory", "--visualize-ego-pose"]
    )
    assert both.visualize_future_trajectory is True and both.visualize_ego_pose is True
    for value in ("0", "-3", "abc"):
        with pytest.raises(SystemExit):
            converter.parse_arguments(required + ["--num-future-trajectory-steps", value])


def test_write_frame_pickle_contains_future_trajectory(tmp_path, monkeypatch):
    converter = _converter()
    monkeypatch.setattr(converter, "_write_frame_images", lambda *args: ({}, {}))
    trajectory = _trajectory()
    (tmp_path / "frame_info_pkl").mkdir()
    converter._write_frame(
        tmp_path,
        "scene",
        0,
        "1000",
        trajectory,
        converter._ego_pose(trajectory, 3),
        {},
        {},
        False,
        4,
    )
    with (tmp_path / "frame_info_pkl" / "1000.pkl").open("rb") as stream:
        frame_data = pickle.load(stream)
    expected_shapes = {
        "rotation": (4, 3, 3),
        "translation": (4, 3),
        "linear_velocity": (4, 3),
        "angular_velocity": (4, 3),
    }
    for name, shape in expected_shapes.items():
        assert frame_data["future_trajectory"][name].shape == shape, name


def test_write_episode_gates_visualizations(tmp_path, monkeypatch):
    converter = _converter()
    monkeypatch.setattr(converter, "_initialize_scene", lambda *args: {})
    monkeypatch.setattr(converter, "_write_frame", lambda *args, **kwargs: "frame.pkl")
    monkeypatch.setattr(converter, "_write_scene_index", lambda *args: None)
    calls: list[str] = []
    monkeypatch.setattr(
        converter, "_ego_pose_visualization", lambda *args: calls.append("ego_pose")
    )
    monkeypatch.setattr(
        converter,
        "_write_future_trajectory_visualizations",
        lambda *args: calls.append("future"),
    )
    trajectory = _trajectory()
    for episode, (visualize_future, visualize_ego) in enumerate(
        [(False, False), (True, False), (False, True), (True, True)]
    ):
        calls.clear()
        converter._write_episode(
            "scene",
            episode,
            tmp_path,
            trajectory,
            ["1000"],
            {},
            False,
            4,
            visualize_future,
            visualize_ego,
        )
        assert (tmp_path / f"sage3d-scene-{episode:06d}").is_dir()
        expected = []
        if visualize_ego:
            expected.append("ego_pose")
        if visualize_future:
            expected.append("future")
        assert calls == expected


def test_navigation_map_artifacts_are_room_masked_and_metadata_is_exact(tmp_path):
    converter = _converter()
    raw = tmp_path / "raw"
    generated = tmp_path / "generated"
    raw.mkdir()
    generated.mkdir()
    Image.fromarray(
        np.array([[255, 0, 255, 255], [127, 255, 0, 0]], dtype=np.uint8)
    ).save(raw / "occupancy.png")
    (raw / "occupancy.json").write_text('{"scale": 0.5, "lower": [10.0, 20.0]}')
    (raw / "structure.json").write_text(
        '{"rooms": [{"profile": [[10.5, 20.0], [11.0, 20.0], '
        '[11.0, 21.0], [10.5, 21.0]]}]}'
    )
    (generated / "map").mkdir()
    np.save(generated / "map/esdf.npy", np.ones((2, 4), dtype=np.float64))
    Image.fromarray(
        np.array([[0, 0, 255, 255], [0, 255, 0, 0]], dtype=np.uint8)
    ).save(generated / "map/safe_mask.png")
    manifest = {
        "scene_dir": str(raw), "robot_radius_m": 0.25, "safety_margin_m": 0.05,
        "map": {"shape": [2, 4], "scale_m_per_pixel": 0.5, "lower_x": 10.0, "lower_y": 20.0, "robot_radius_m": 0.25, "safety_margin_m": 0.05, "required_path_clearance_m": 0.3, "safe_mask_semantics": "robot_footprint_v1"},
    }
    (tmp_path / "out").mkdir()
    info = converter._write_navigation_map(tmp_path / "out", "scene", generated, manifest)
    occupancy = np.asarray(Image.open(tmp_path / "out/map/occupancy.png"))
    assert np.isin(occupancy, (0, 127, 255)).all()
    assert occupancy[0, 3] == 255 and occupancy[0, 0] == 127
    assert info["pixel_to_world"] == [[-0.5, 0, 11.75], [0, 0.5, 20.25], [0, 0, 1]]
    assert np.load(tmp_path / "out/map/clearance.npy").dtype == np.float32
    assert (tmp_path / "out/map/traversability.png").is_file()

    (tmp_path / "out/frame_info_pkl").mkdir()
    trajectory = _trajectory()
    converter._write_frame(
        tmp_path / "out", "scene", 0, "1000", trajectory,
        converter._ego_pose(trajectory, 3), {}, {"navigation_map_2d": info}, False, 2,
    )
    with (tmp_path / "out/frame_info_pkl/1000.pkl").open("rb") as stream:
        frame = pickle.load(stream)
    assert frame["scene_info"]["navigation_map_2d"] == info
    converter._write_scene_index(tmp_path / "out", "scene", {"1000": "scene/frame_info_pkl/1000.pkl"})
    with (tmp_path / "out/info.pkl").open("rb") as stream:
        assert pickle.load(stream)["scene"]["scene_info"] == {}


def test_future_trajectory_visualization_writes_one_png_per_frame(tmp_path):
    converter = _converter()
    frame_ids = ["1000", "1001", "1002", "1003"]
    converter._write_future_trajectory_visualizations(tmp_path, _trajectory(), frame_ids, 6)
    pngs = sorted((tmp_path / "future_trajectory_vis").iterdir())
    assert [path.name for path in pngs] == [f"{frame_id}.png" for frame_id in frame_ids]
    for png in pngs:
        with Image.open(png) as image:
            assert image.size[0] > 0 and image.size[1] > 0
            assert np.asarray(image).size > 0


def test_future_trajectory_figure_uses_relative_coordinates():
    converter = _converter()
    trajectory = _trajectory()
    ego = converter._ego_pose(trajectory, 0)
    goal = converter._goal(ego, converter._ego_pose(trajectory, 3))
    future = converter._future_trajectory(trajectory, 0, 2)
    fig = converter._future_trajectory_figure(ego, goal, future)
    try:
        ax = fig.axes[0]
        lines = {line.get_label(): line for line in ax.lines}
        axis = converter._EGO_AXIS_LENGTH_M
        scale = converter._EGO_VELOCITY_SCALE_S

        velocity = lines["ego_velocity"].get_xydata()
        assert np.allclose(velocity[1], scale * ego["linear_velocity"][:2], atol=1e-6)
        goal_heading = lines["goal_heading"].get_xydata()
        assert np.allclose(goal_heading[0], goal["translation"][:2], atol=1e-6)
        assert np.allclose(
            goal_heading[1],
            goal_heading[0] + axis * goal["rotation"][:2, 0],
            atol=1e-6,
        )

        base = future["translation"][0][:2]
        future_omega = float(future["angular_velocity"][0][2])
        expected = {
            "future_x": axis * future["rotation"][0][:2, 0],
            "future_y": axis * future["rotation"][0][:2, 1],
            "future_velocity": scale * future["linear_velocity"][0][:2],
            "future_angular": axis
            * (future["rotation"][0][:2, :2] @ np.array([np.cos(future_omega), np.sin(future_omega)])),
        }
        colors = {"future_x": "red", "future_y": "green", "future_velocity": "blue", "future_angular": "purple"}
        for name, vector in expected.items():
            segment = lines[name].get_xydata()
            assert np.allclose(segment[0], base, atol=1e-6)
            assert np.allclose(segment[1], base + vector, atol=1e-6)
            assert lines[name].get_color() == colors[name]
            assert lines[name].get_alpha() == 0.2
            assert lines[name].get_linewidth() == 0.5

        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(legend_labels) >= set(expected)
        for name in expected:
            assert legend_labels.count(name) == 1
        assert float(ax.get_aspect()) == 1.0
        assert ax.get_xlabel() == "X forward [m]"
        assert ax.get_ylabel() == "Y left [m]"
    finally:
        plt.close(fig)
