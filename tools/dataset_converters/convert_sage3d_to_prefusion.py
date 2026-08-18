"""Convert validated Sage3D episodes to independent prefusion scenes."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import cv2

_EGO_AXIS_LENGTH_M = 0.2
_EGO_VELOCITY_SCALE_S = 1.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene-root", type=Path, required=True)
    parser.add_argument("--output-scene-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--clone-camera-images", action="store_true", default=False)
    parser.add_argument("--num-future-trajectory-steps", type=_positive_int, default=20)
    parser.add_argument(
        "--visualize-future-trajectory", action="store_true", default=False
    )
    parser.add_argument("--visualize-ego-pose", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    input_root = args.input_scene_root
    output_root = args.output_scene_root
    if not input_root.is_dir():
        logger.error("input scene root is not a directory: {}", input_root)
        return 2
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        scene_dirs = _select_scene_dirs(input_root, args.scene_ids)
        if not scene_dirs:
            logger.error("no source scenes selected")
            return 2
        succeeded, skipped, existing, reasons = _convert_scenes(
            scene_dirs,
            output_root,
            args.clone_camera_images,
            args.num_future_trajectory_steps,
            args.visualize_future_trajectory,
            args.visualize_ego_pose,
        )
    except Exception as error:
        logger.error("fatal conversion error: {}", error)
        return 2

    _report_summary(len(scene_dirs), succeeded, skipped, reasons)
    return 0 if succeeded > 0 or existing > 0 else 1


def _report_summary(
    source_scene_count: int,
    succeeded: int,
    skipped: int,
    reasons: Counter[str],
) -> None:
    reason_summary = (
        ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())) or "none"
    )
    logger.info(
        "summary source_scenes={} successful_episodes={} skipped={} reasons={}",
        source_scene_count,
        succeeded,
        skipped,
        reason_summary,
    )


def _select_scene_dirs(input_root: Path, scene_ids: list[str] | None) -> list[Path]:
    if scene_ids is not None:
        return [input_root / scene_id for scene_id in scene_ids]
    return sorted(
        (path for path in input_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


def _convert_scenes(
    scene_dirs: list[Path],
    output_root: Path,
    clone_images: bool,
    num_future_trajectory_steps: int,
    visualize_future_trajectory: bool,
    visualize_ego_pose: bool,
) -> tuple[int, int, int, Counter[str]]:
    reasons: Counter[str] = Counter()
    succeeded = skipped = existing = 0
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            _warning(scene_dir.name, None, None, "source scene directory is missing")
            reasons["missing source scene"] += 1
            skipped += 1
            continue
        scene_succeeded, scene_skipped, scene_existing = _convert_scene(
            scene_dir,
            output_root,
            clone_images,
            reasons,
            num_future_trajectory_steps,
            visualize_future_trajectory,
            visualize_ego_pose,
        )
        succeeded += scene_succeeded
        skipped += scene_skipped
        existing += scene_existing
    return succeeded, skipped, existing, reasons


def _convert_scene(
    scene_dir: Path,
    output_root: Path,
    clone_images: bool,
    reasons: Counter[str],
    num_future_trajectory_steps: int,
    visualize_future_trajectory: bool,
    visualize_ego_pose: bool,
) -> tuple[int, int, int]:
    source_scene_id = scene_dir.name
    try:
        profiles, control_dt, records, validated, manifest = _load_scene_inputs(
            scene_dir
        )
    except Exception as error:
        _warning(source_scene_id, None, None, str(error))
        reasons["invalid source scene"] += 1
        return 0, 1, 0

    succeeded = skipped = existing = 0
    for record in tqdm(records, desc=f"processing {source_scene_id}"):
        episode: int | None = None
        try:
            episode = int(record["episode_index"])
            created = _convert_episode(
                scene_dir,
                episode,
                record,
                validated,
                profiles,
                control_dt,
                output_root,
                clone_images,
                num_future_trajectory_steps,
                visualize_future_trajectory,
                visualize_ego_pose,
                manifest=manifest,
            )
            if created:
                succeeded += 1
            else:
                _warning(source_scene_id, episode, None, "target scene already exists")
                reasons["target exists"] += 1
                skipped += 1
                existing += 1
        except Exception as error:
            message = str(error)
            camera_match = re.search(r"camera=([^:]+):", message)
            _warning(
                source_scene_id,
                episode,
                camera_match.group(1) if camera_match else None,
                message,
            )
            reasons["invalid episode"] += 1
            skipped += 1
    return succeeded, skipped, existing


def _convert_episode(
    scene_dir: Path,
    episode: int,
    record: dict[str, Any],
    validated: dict[int, dict[str, Any]],
    profiles: dict[str, Any],
    control_dt: float,
    output_root: Path,
    clone_images: bool,
    num_future_trajectory_steps: int,
    visualize_future_trajectory: bool,
    visualize_ego_pose: bool,
    manifest: dict[str, Any],
) -> bool:
    source_scene_id = scene_dir.name
    expected_name, t_output = _validate_episode_record(record, episode, validated)
    npz_path = scene_dir / "optimized_trajectories" / expected_name
    target = output_root / f"sage3d-{source_scene_id}-{episode:06d}"
    if target.exists():
        return False

    trajectory = _trajectory(npz_path, control_dt, t_output)
    frame_ids = _frame_ids(npz_path, len(trajectory["time_s"]), control_dt)
    cameras = _load_cameras(
        scene_dir / "rendered", source_scene_id, episode, len(frame_ids), profiles
    )
    _write_episode(
        source_scene_id,
        episode,
        output_root,
        trajectory,
        frame_ids,
        cameras,
        clone_images,
        num_future_trajectory_steps,
        visualize_future_trajectory,
        visualize_ego_pose,
        source_scene_dir=scene_dir,
        manifest=manifest,
    )
    return True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _warning(scene: str, episode: int | None, camera: str | None, reason: str) -> None:
    logger.warning(
        "source_scene={} episode={} camera={} reason={}",
        scene,
        "-" if episode is None else f"{episode:06d}",
        camera or "-",
        reason,
    )


def _same(actual: Any, expected: Any, name: str, *, atol: float = 1e-5) -> None:
    try:
        actual_array = np.asarray(actual, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        if actual != expected:
            raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")
        return
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, rtol=1e-5, atol=atol
    ):
        raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")


def _profile_resolution(profile: dict[str, Any]) -> tuple[int, int]:
    resolution = profile.get("resolution")
    if isinstance(resolution, dict):
        resolution = [resolution.get("width"), resolution.get("height")]
    if resolution is None:
        resolution = [profile.get("width"), profile.get("height")]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("camera profile resolution must be [width, height]")
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError("camera profile resolution must be positive")
    return width, height


def _profile_model(profile: dict[str, Any]) -> str:
    model = profile.get("model", profile.get("camera_model"))
    if model not in {"pinhole", "opencv_fisheye"}:
        raise ValueError(f"unsupported camera model {model!r}")
    return str(model)


def _profile_intrinsic(profile: dict[str, Any]) -> list[float]:
    width, height = _profile_resolution(profile)
    model = _profile_model(profile)
    coefficients = profile.get(
        "fisheye_coefficients", profile.get("distortion_coefficients")
    )
    if model == "opencv_fisheye" and (
        not isinstance(coefficients, (list, tuple)) or len(coefficients) != 4
    ):
        raise ValueError("opencv_fisheye profile requires four coefficients")
    focal = profile.get("focal_length_pixels")
    if focal is None:
        focal = profile.get("focal_length_px")
    if focal is None and model == "pinhole":
        hfov = float(profile["horizontal_fov_deg"])
        focal = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    if focal is None and model == "opencv_fisheye":
        k1, k2, k3, k4 = map(float, coefficients)
        theta = math.radians(float(profile["horizontal_fov_deg"])) / 2.0
        theta2 = theta * theta
        theta_d = theta * (
            1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )
        focal = (width / 2.0) / theta_d
    if isinstance(focal, (list, tuple)):
        fx, fy = map(float, focal)
    elif focal is not None:
        fx = fy = float(focal)
    else:
        raise ValueError("camera profile lacks focal length")
    if fx <= 0 or fy <= 0 or not np.isclose(fx, fy):
        raise ValueError(
            "camera profile must have finite positive square-pixel focal length"
        )
    intrinsic = [width / 2.0, height / 2.0, fx, fy]
    if model == "opencv_fisheye":
        intrinsic.extend(float(value) for value in coefficients)
    if not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic contains non-finite values")
    return intrinsic


def _profile_extrinsic(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    extrinsic = profile.get("extrinsic")
    if not isinstance(extrinsic, dict):
        raise ValueError("camera profile lacks extrinsic")
    translation = np.asarray(extrinsic.get("translation_body_m"), dtype=np.float32)
    rpy = np.asarray(extrinsic.get("rotation_rpy_deg"), dtype=np.float32)
    if (
        translation.shape != (3,)
        or rpy.shape != (3,)
        or not np.isfinite([translation, rpy]).all()
    ):
        raise ValueError("camera extrinsic must contain finite xyz translation and RPY")
    rotation = (
        Rotation.from_euler("xyz", rpy, degrees=True).as_matrix().astype(np.float32)
    )
    return rotation, translation


def _calibration(profile: dict[str, Any]) -> dict[str, Any]:
    rotation, translation = _profile_extrinsic(profile)
    return {
        "camera_type": "PerspectiveCamera"
        if _profile_model(profile) == "pinhole"
        else "FisheyeCamera",
        "intrinsic": np.asarray(_profile_intrinsic(profile), dtype=np.float32),
        "extrinsic": (rotation, translation),
    }


def _validate_summary(
    summary: dict[str, Any],
    profile: dict[str, Any],
    source_scene_id: str,
    camera_id: str,
    mode: str,
) -> None:
    if str(summary.get("scene_id")) != source_scene_id:
        raise ValueError(f"{mode} summary scene_id mismatch")
    if summary.get("camera_id") != camera_id:
        raise ValueError(f"{mode} summary camera_id mismatch")
    if summary.get("render_mode") != mode:
        raise ValueError(f"{mode} summary render_mode mismatch")
    model = _profile_model(profile)
    _same(summary.get("camera_model"), model, f"{mode} camera model")
    _same(
        summary.get("resolution"),
        list(_profile_resolution(profile)),
        f"{mode} resolution",
    )
    intrinsic = _profile_intrinsic(profile)
    _same(summary.get("principal_point"), intrinsic[:2], f"{mode} principal point")
    _same(summary.get("focal_length_pixels"), intrinsic[2], f"{mode} focal length")
    summary_extrinsic = summary.get("camera_extrinsic")
    profile_extrinsic = profile.get("extrinsic")
    _same(
        summary_extrinsic.get("translation_body_m"),
        profile_extrinsic.get("translation_body_m"),
        f"{mode} extrinsic translation",
    )
    _same(
        summary_extrinsic.get("rotation_rpy_deg"),
        profile_extrinsic.get("rotation_rpy_deg"),
        f"{mode} extrinsic rotation",
    )
    if model == "opencv_fisheye":
        _same(
            summary.get("fisheye_coefficients"),
            intrinsic[4:],
            f"{mode} fisheye coefficients",
        )
    if mode == "depth":
        if summary.get("depth_type") != "distance_to_camera":
            raise ValueError(f"{mode} summary depth_type is not distance_to_camera")
        depth_scale = float(summary.get("depth_scale", 0))
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError(f"{mode} summary depth_scale must be positive")


def _metadata_episode_records(
    metadata: dict[str, Any], name: str, *, require_list: bool = False
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    records = metadata.get("episodes", [])
    if isinstance(records, dict):
        if require_list:
            raise ValueError(f"{name} episodes is not a list")
        records = records.values()
    elif not isinstance(records, list):
        raise ValueError(f"{name} episodes is not a list or object")
    records = list(records)
    indexed = {}
    for record in records:
        if isinstance(record, dict) and "episode_index" in record:
            try:
                episode = int(record["episode_index"])
            except (TypeError, ValueError, OverflowError):
                continue
            if episode in indexed:
                raise ValueError(f"duplicate {name} episode_index {episode}")
            indexed[episode] = record
    return records, indexed


def _frame_files(
    directory: Path, episode: int, suffix: str, frame_count: int
) -> list[Path]:
    pattern = re.compile(
        rf"^episode_{episode:06d}_(\d+)\.{re.escape(suffix)}$", re.IGNORECASE
    )
    indexed: dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"duplicate {suffix} frame index {index}")
            indexed[index] = path
    expected = set(range(frame_count))
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{suffix} frame indices mismatch; missing={missing} extra={extra}"
        )
    return [indexed[index] for index in range(frame_count)]


def _validate_image(path: Path, width: int, height: int, *, depth: bool) -> None:
    with Image.open(path) as image:
        array = np.asarray(image)
    if depth:
        if array.shape != (height, width) or array.dtype != np.uint16:
            raise ValueError(
                f"depth must be 2D uint16, got shape={array.shape} dtype={array.dtype}"
            )
    elif array.shape != (height, width, 3) or array.dtype != np.uint8:
        raise ValueError(
            f"RGB must be HxWx3 uint8, got shape={array.shape} dtype={array.dtype}"
        )


def _trajectory(
    npz_path: Path, control_dt: float, t_output: float
) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        required = ("time_s", "pose_world", "velocity_world_mps", "yaw_rate_radps")
        arrays = {key: np.asarray(archive[key]) for key in required}
    time_s = arrays["time_s"]
    if time_s.ndim != 1 or len(time_s) == 0 or not np.isfinite(time_s).all():
        raise ValueError("time_s must be a non-empty finite 1D array")
    frame_count = len(time_s)
    if not np.isclose(time_s[0], 0.0, atol=1e-8):
        raise ValueError("time_s must start at zero")
    if frame_count > 1 and not np.allclose(
        np.diff(time_s), control_dt, rtol=1e-6, atol=1e-8
    ):
        raise ValueError("time_s does not have the configured fixed interval")
    if not np.isclose(time_s[-1], t_output, rtol=1e-6, atol=1e-8):
        raise ValueError("time_s endpoint does not match candidate T_output")
    if arrays["pose_world"].shape != (frame_count, 3):
        raise ValueError("pose_world must have shape [K, 3]")
    if arrays["velocity_world_mps"].shape != (frame_count, 2):
        raise ValueError("velocity_world_mps must have shape [K, 2]")
    if arrays["yaw_rate_radps"].shape != (frame_count,):
        raise ValueError("yaw_rate_radps must have shape [K]")
    for key in ("pose_world", "velocity_world_mps", "yaw_rate_radps"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    return arrays


def _frame_ids(npz_path: Path, frame_count: int, control_dt: float) -> list[str]:
    initial_ms = npz_path.stat().st_mtime_ns // 1_000_000
    step_ms = control_dt * 1000.0
    ids = [
        str(int(round(initial_ms + index * step_ms))) for index in range(frame_count)
    ]
    integers = list(map(int, ids))
    if any(b <= a for a, b in zip(integers, integers[1:])):
        raise ValueError(
            "control_dt_s does not produce unique strictly increasing millisecond frame IDs"
        )
    return ids


def _validate_mask(path: Path, width: int, height: int) -> None:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"mask must be grayscale mode L, got {image.mode!r}")
        array = np.asarray(image)
    if array.shape != (height, width):
        raise ValueError(f"mask resolution {array.shape[::-1]} != {(width, height)}")
    if not ((array == 0) | (array == 255)).all():
        raise ValueError("mask must contain only 0 and 255")


def _copy_or_link(source: Path, destination: Path, clone: bool) -> None:
    if clone:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def _initialize_scene(
    temporary: Path,
    scene_name: str,
    cameras: dict[str, dict[str, Any]],
    navigation_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    (temporary / "frame_info_pkl").mkdir()
    for camera_id in cameras:
        (temporary / "camera_image" / camera_id).mkdir(parents=True)
        (temporary / "camera_image_depth" / camera_id).mkdir(parents=True)
    (temporary / "self_mask").mkdir()

    calibrations = {
        camera_id: camera["calibration"] for camera_id, camera in cameras.items()
    }
    masks = {}
    for camera_id, camera in cameras.items():
        mask_relative = Path(scene_name) / "self_mask" / f"{camera_id}.png"
        masks[camera_id] = str(mask_relative)
        shutil.copy2(camera["mask_file"], temporary / "self_mask" / f"{camera_id}.png")
    scene_info = {"camera_mask": masks, "calibration": calibrations}
    if navigation_map is not None:
        scene_info["navigation_map_2d"] = navigation_map
    return scene_info


def _write_navigation_map(
    temporary: Path, scene_name: str, source_scene_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    raw_scene = Path(manifest["scene_dir"])
    with (raw_scene / "occupancy.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    occupancy_path = raw_scene / "occupancy.png"
    occupancy = np.asarray(Image.open(occupancy_path).convert("L"))
    height, width = occupancy.shape
    scale = float(metadata["scale"])
    lower_x, lower_y = map(float, metadata["lower"][:2])
    map_info = manifest.get("map")
    if not isinstance(map_info, dict):
        raise ValueError("trajectory manifest lacks map metadata")
    if (
        map_info.get("shape") != [height, width]
        or not np.isclose(float(map_info["scale_m_per_pixel"]), scale)
        or not np.isclose(float(map_info["lower_x"]), lower_x)
        or not np.isclose(float(map_info["lower_y"]), lower_y)
    ):
        raise ValueError("manifest map metadata does not match occupancy metadata")
    if (
        scale <= 0
        or width == 0
        or height == 0
        or not np.isin(occupancy, (0, 127, 255)).all()
    ):
        raise ValueError("invalid InteriorGS occupancy map")
    with (raw_scene / "structure.json").open(encoding="utf-8") as stream:
        structure = json.load(stream)
    room_mask = np.zeros((height, width), dtype=np.uint8)
    rooms = 0
    for room in structure.get("rooms", []):
        profile = room.get("profile", [])
        if len(profile) < 3:
            continue
        pixels = np.asarray(
            [
                [
                    width - 1 - int(round((float(x) - lower_x) / scale - 0.5)),
                    int(round((float(y) - lower_y) / scale - 0.5)),
                ]
                for x, y in profile
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(room_mask, [pixels], 1)
        rooms += 1
    if not rooms:
        raise ValueError("structure.json has no valid room polygons")
    mask = room_mask.astype(bool)
    occupancy = np.where(mask, occupancy, 127).astype(np.uint8)
    generated_map = source_scene_dir / "map"
    clearance = np.load(generated_map / "esdf.npy", allow_pickle=False)
    traversability = np.asarray(
        Image.open(generated_map / "safe_mask.png").convert("L")
    )
    if clearance.dtype.kind != "f":
        raise ValueError("esdf.npy clearance must be floating")
    if (
        clearance.ndim != 2
        or clearance.shape != occupancy.shape
        or not np.isfinite(clearance).all()
        or (clearance < 0).any()
    ):
        raise ValueError(
            "esdf.npy clearance must be finite, nonnegative, and shape-matched"
        )
    if traversability.shape != occupancy.shape:
        raise ValueError("safe_mask.png shape must match occupancy")
    if not np.isin(traversability, (0, 255)).all():
        raise ValueError("safe_mask.png must be binary")
    robot_radius = float(manifest["robot_radius_m"])
    safety_margin = float(manifest["safety_margin_m"])
    threshold = float(map_info["required_path_clearance_m"])
    if map_info.get("safe_mask_semantics") != "robot_footprint_v1":
        raise ValueError("manifest map safe_mask_semantics must be robot_footprint_v1")
    if (
        not np.isclose(float(map_info["robot_radius_m"]), robot_radius)
        or not np.isclose(float(map_info["safety_margin_m"]), safety_margin)
        or not np.isclose(threshold, robot_radius + safety_margin)
    ):
        raise ValueError("manifest map radius, margin, or clearance threshold mismatch")
    expected_traversability = ((occupancy == 255) & (clearance >= threshold)).astype(
        np.uint8
    ) * 255
    if not np.array_equal(traversability, expected_traversability):
        raise ValueError(
            "safe_mask.png does not match room-masked occupancy and clearance threshold"
        )
    (temporary / "map").mkdir()
    Image.fromarray(occupancy).save(temporary / "map" / "occupancy.png")
    np.save(
        temporary / "map" / "clearance.npy", np.asarray(clearance, dtype=np.float32)
    )
    Image.fromarray(traversability).save(temporary / "map" / "traversability.png")
    prefix = Path(scene_name) / "map"
    return {
        "occupancy_path": str(prefix / "occupancy.png"),
        "clearance_path": str(prefix / "clearance.npy"),
        "traversability_path": str(prefix / "traversability.png"),
        "shape": [height, width],
        "resolution": scale,
        "pixel_to_world": [
            [-scale, 0, lower_x + (width - 0.5) * scale],
            [0, scale, lower_y + 0.5 * scale],
            [0, 0, 1],
        ],
        "occupancy_encoding": {"unknown": 127, "free": 255, "occupied": 0},
        "clearance_semantics": "unsigned meters",
        "traversability_robot_radius_m": robot_radius,
        "traversability_safety_margin_m": safety_margin,
        "traversability_threshold_m": threshold,
    }


def _write_frame_images(
    temporary: Path,
    scene_name: str,
    frame_id: str,
    frame_index: int,
    cameras: dict[str, dict[str, Any]],
    clone_images: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    camera_images = {}
    depth_images = {}
    for camera_id, camera in cameras.items():
        rgb_name = f"{frame_id}.jpg"
        depth_name = f"{frame_id}.npz"
        rgb_relative = Path(scene_name) / "camera_image" / camera_id / rgb_name
        depth_relative = (
            Path(scene_name) / "camera_image_depth" / camera_id / depth_name
        )
        _copy_or_link(
            camera["rgb_files"][frame_index],
            temporary / "camera_image" / camera_id / rgb_name,
            clone_images,
        )
        with Image.open(camera["depth_files"][frame_index]) as image:
            depth = (
                np.asarray(image, dtype=np.uint16).astype(np.float32)
                / camera["depth_scale"]
            )
        np.savez_compressed(
            temporary / "camera_image_depth" / camera_id / depth_name, depth=depth
        )
        camera_images[camera_id] = str(rgb_relative)
        depth_images[camera_id] = str(depth_relative)
    return camera_images, depth_images


def _ego_pose(
    trajectory: dict[str, np.ndarray], frame_index: int
) -> dict[str, np.ndarray]:
    x, y, yaw = trajectory["pose_world"][frame_index]
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    velocity_world = np.asarray(
        [*trajectory["velocity_world_mps"][frame_index], 0.0], dtype=np.float32
    )
    angular_world = np.asarray(
        [0.0, 0.0, trajectory["yaw_rate_radps"][frame_index]], dtype=np.float32
    )
    ego_pose = {
        "rotation": rotation,
        "translation": np.asarray([x, y, 0.0], dtype=np.float32),
        "linear_velocity": rotation.T @ velocity_world,
        "angular_velocity": rotation.T @ angular_world,
    }
    if not all(np.isfinite(value).all() for value in ego_pose.values()):
        raise ValueError(f"frame {frame_index} ego pose contains non-finite values")
    return ego_pose


def _goal(
    ego_pose: dict[str, np.ndarray], terminal_ego_pose: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    relative_rotation = ego_pose["rotation"].T @ terminal_ego_pose["rotation"]
    return {
        "rotation": relative_rotation,
        "translation": ego_pose["rotation"].T
        @ (terminal_ego_pose["translation"] - ego_pose["translation"]),
        "linear_velocity": relative_rotation @ terminal_ego_pose["linear_velocity"],
        "angular_velocity": relative_rotation @ terminal_ego_pose["angular_velocity"],
    }


def _future_trajectory(
    trajectory: dict[str, np.ndarray],
    frame_index: int,
    num_future_trajectory_steps: int,
) -> dict[str, np.ndarray]:
    terminal_index = len(trajectory["pose_world"]) - 1
    ego_pose = _ego_pose(trajectory, frame_index)
    trajectory_steps = [
        _goal(
            ego_pose, _ego_pose(trajectory, min(frame_index + offset, terminal_index))
        )
        for offset in range(1, num_future_trajectory_steps + 1)
    ]
    fields = ("rotation", "translation", "linear_velocity", "angular_velocity")
    return {
        field: np.stack([step[field] for step in trajectory_steps], axis=0)
        for field in fields
    }


def _write_frame(
    temporary: Path,
    scene_name: str,
    frame_index: int,
    frame_id: str,
    trajectory: dict[str, np.ndarray],
    terminal_ego_pose: dict[str, np.ndarray],
    cameras: dict[str, dict[str, Any]],
    scene_info: dict[str, Any],
    clone_images: bool,
    num_future_trajectory_steps: int,
) -> str:
    camera_images, depth_images = _write_frame_images(
        temporary, scene_name, frame_id, frame_index, cameras, clone_images
    )
    ego_pose = _ego_pose(trajectory, frame_index)
    frame_data = {
        "camera_image": camera_images,
        "camera_image_depth": depth_images,
        "ego_pose": ego_pose,
        "goal": _goal(ego_pose, terminal_ego_pose),
        "future_trajectory": _future_trajectory(
            trajectory, frame_index, num_future_trajectory_steps
        ),
        "scene_info": scene_info,
    }
    frame_name = f"{frame_id}.pkl"
    with (temporary / "frame_info_pkl" / frame_name).open("wb") as stream:
        pickle.dump(frame_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return str(Path(scene_name) / "frame_info_pkl" / frame_name)


def _write_scene_index(
    temporary: Path, scene_name: str, frame_index: dict[str, str]
) -> None:
    info = {scene_name: {"scene_info": {}, "frame_info": frame_index}}
    with (temporary / "info.pkl").open("wb") as stream:
        pickle.dump(info, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _future_trajectory_figure(
    ego_pose: dict[str, np.ndarray],
    goal: dict[str, np.ndarray],
    future_trajectory: dict[str, np.ndarray],
) -> plt.Figure:
    """Build a 2D top-down body-frame figure of ego, goal, and future trajectory."""
    axis_length = _EGO_AXIS_LENGTH_M
    velocity_scale = _EGO_VELOCITY_SCALE_S
    xs: list[float] = [0.0]
    ys: list[float] = [0.0]

    fig, ax = plt.subplots(figsize=(8, 6))

    def segment(origin, vector, *args, **kwargs):
        start = np.asarray(origin, dtype=np.float64)[:2]
        end = start + np.asarray(vector, dtype=np.float64)[:2]
        ax.plot([start[0], end[0]], [start[1], end[1]], *args, **kwargs)
        xs.extend([start[0], end[0]])
        ys.extend([start[1], end[1]])

    segment((0.0, 0.0), (axis_length, 0.0), color="red", linewidth=2, label="ego_x")
    segment((0.0, 0.0), (0.0, axis_length), color="green", linewidth=2, label="ego_y")
    segment(
        (0.0, 0.0),
        velocity_scale * ego_pose["linear_velocity"][:2],
        color="blue",
        linewidth=2,
        label="ego_velocity",
    )
    omega = float(ego_pose["angular_velocity"][2])
    segment(
        (0.0, 0.0),
        axis_length * np.array([math.cos(omega), math.sin(omega)]),
        color="purple",
        linewidth=2,
        label="ego_angular",
    )

    goal_position = goal["translation"][:2]
    xs.append(goal_position[0])
    ys.append(goal_position[1])
    ax.plot(
        [goal_position[0]],
        [goal_position[1]],
        marker="*",
        color="magenta",
        markersize=14,
        label="goal",
    )
    segment(
        goal_position,
        axis_length * goal["rotation"][:2, 0],
        color="magenta",
        linewidth=2,
        label="goal_heading",
    )

    for index, (translation, rotation, linear, angular) in enumerate(
        zip(
            future_trajectory["translation"],
            future_trajectory["rotation"],
            future_trajectory["linear_velocity"],
            future_trajectory["angular_velocity"],
        )
    ):
        position = translation[:2]
        xs.append(position[0])
        ys.append(position[1])
        ax.plot([position[0]], [position[1]], marker=".", color="grey", alpha=0.2)
        future_omega = float(angular[2])
        angular_direction = rotation[:2, :2] @ np.array(
            [math.cos(future_omega), math.sin(future_omega)]
        )
        for label, vector, color in (
            ("future_x", axis_length * rotation[:2, 0], "red"),
            ("future_y", axis_length * rotation[:2, 1], "green"),
            ("future_velocity", velocity_scale * linear[:2], "blue"),
            ("future_angular", axis_length * angular_direction, "purple"),
        ):
            segment(
                position,
                vector,
                color=color,
                linewidth=0.5,
                alpha=0.2,
                label=label if index == 0 else "_nolegend_",
            )

    margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1 + axis_length
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X forward [m]")
    ax.set_ylabel("Y left [m]")
    ax.set_title("future trajectory")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def _write_future_trajectory_visualizations(
    temporary: Path,
    trajectory: dict[str, np.ndarray],
    frame_ids: list[str],
    num_future_trajectory_steps: int,
) -> None:
    (temporary / "future_trajectory_vis").mkdir()
    terminal_ego_pose = _ego_pose(trajectory, len(frame_ids) - 1)
    for frame_index, frame_id in enumerate(frame_ids):
        ego_pose = _ego_pose(trajectory, frame_index)
        future_trajectory = _future_trajectory(
            trajectory, frame_index, num_future_trajectory_steps
        )
        fig = _future_trajectory_figure(
            ego_pose, _goal(ego_pose, terminal_ego_pose), future_trajectory
        )
        try:
            fig.savefig(
                temporary / "future_trajectory_vis" / f"{frame_id}.png", dpi=150
            )
        finally:
            plt.close(fig)


def _ego_pose_visualization(
    temporary: Path, trajectory: dict[str, np.ndarray], frame_ids: list[str]
) -> go.Figure:
    """Write a self-contained interactive ego-pose animation into temporary.

    Returns the built figure for inspection.
    """
    pose = trajectory["pose_world"]
    time_s = trajectory["time_s"]
    frame_count = len(frame_ids)

    def per_frame_traces(frame_index: int) -> list[go.Scatter]:
        ego = _ego_pose(trajectory, frame_index)
        r = ego["rotation"]
        t = ego["translation"]
        yaw = float(pose[frame_index, 2])
        origin_x, origin_y = float(t[0]), float(t[1])
        body_x = (
            origin_x + _EGO_AXIS_LENGTH_M * float(r[0, 0]),
            origin_y + _EGO_AXIS_LENGTH_M * float(r[1, 0]),
        )
        body_y = (
            origin_x + _EGO_AXIS_LENGTH_M * float(r[0, 1]),
            origin_y + _EGO_AXIS_LENGTH_M * float(r[1, 1]),
        )
        velocity = r[:2, :2] @ ego["linear_velocity"][:2]
        velocity_end = (
            origin_x + _EGO_VELOCITY_SCALE_S * float(velocity[0]),
            origin_y + _EGO_VELOCITY_SCALE_S * float(velocity[1]),
        )
        angular_z = float(ego["angular_velocity"][2])
        angular_end = (
            origin_x + _EGO_AXIS_LENGTH_M * math.cos(yaw + angular_z),
            origin_y + _EGO_AXIS_LENGTH_M * math.sin(yaw + angular_z),
        )
        text = (
            f"frame_id={frame_ids[frame_index]}<br>"
            f"t={float(time_s[frame_index]):.3f} s<br>"
            f"translation=({origin_x:.3f}, {origin_y:.3f}, {float(t[2]):.3f}) m<br>"
            f"yaw={yaw:.3f} rad<br>"
            f"linear_velocity_body=("
            f"{float(ego['linear_velocity'][0]):.3f}, {float(ego['linear_velocity'][1]):.3f}) m/s<br>"
            f"angular_velocity_z={angular_z:.3f} rad/s"
        )

        def line(x0, y0, x1, y1, color, name) -> go.Scatter:
            return go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=3),
                customdata=[text, text],
                hovertemplate="%{customdata}<extra></extra>",
                name=name,
            )

        return [
            go.Scatter(
                x=[origin_x],
                y=[origin_y],
                mode="markers",
                marker=dict(color="black", size=7),
                customdata=[text],
                hovertemplate="%{customdata}<extra></extra>",
                name="origin",
            ),
            line(origin_x, origin_y, *body_x, "red", "body_x"),
            line(origin_x, origin_y, *body_y, "green", "body_y"),
            line(origin_x, origin_y, *velocity_end, "blue", "velocity"),
            line(origin_x, origin_y, *angular_end, "purple", "angular_axis"),
        ]

    trajectory_trace = go.Scatter(
        x=pose[:, 0].tolist(),
        y=pose[:, 1].tolist(),
        mode="lines",
        line=dict(color="grey", width=1.5),
        hoverinfo="skip",
        name="trajectory",
    )

    fig = go.Figure(data=[trajectory_trace, *per_frame_traces(0)])
    fig.frames = [
        go.Frame(
            name=frame_ids[index],
            data=per_frame_traces(index),
            traces=[1, 2, 3, 4, 5],
        )
        for index in range(frame_count)
    ]

    xmin, xmax = float(np.min(pose[:, 0])), float(np.max(pose[:, 0]))
    ymin, ymax = float(np.min(pose[:, 1])), float(np.max(pose[:, 1]))
    max_speed = float(np.max(np.linalg.norm(trajectory["velocity_world_mps"], axis=1)))
    margin = max(_EGO_AXIS_LENGTH_M, _EGO_VELOCITY_SCALE_S * max_speed)
    span = max(xmax - xmin, ymax - ymin) / 2.0 + margin
    xmid, ymid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    if frame_count > 1:
        duration_ms = int(round(float(time_s[1] - time_s[0]) * 1000.0))
    else:
        duration_ms = 100

    fig.update_layout(
        title="Ego-pose visualization",
        xaxis=dict(range=[xmid - span, xmid + span], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[ymid - span, ymid + span]),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.0,
                y=1.15,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=duration_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                y=0.0,
                xanchor="left",
                x=0.0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [frame_ids[index]],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                        label=str(index),
                    )
                    for index in range(frame_count)
                ],
            )
        ],
    )

    html = fig.to_html(full_html=True, include_plotlyjs=True, auto_play=False)
    with zipfile.ZipFile(
        temporary / "ego_pose_visualization.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("ego_pose_visualization.html", html)
    return fig


def _write_episode(
    source_scene_id: str,
    episode: int,
    output_root: Path,
    trajectory: dict[str, np.ndarray],
    frame_ids: list[str],
    cameras: dict[str, dict[str, Any]],
    clone_images: bool,
    num_future_trajectory_steps: int,
    visualize_future_trajectory: bool,
    visualize_ego_pose: bool,
    *,
    source_scene_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> str:
    scene_name = f"sage3d-{source_scene_id}-{episode:06d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".{scene_name}.", dir=output_root))
    try:
        navigation_map = (
            _write_navigation_map(temporary, scene_name, source_scene_dir, manifest)
            if source_scene_dir is not None and manifest is not None
            else None
        )
        scene_info = _initialize_scene(temporary, scene_name, cameras, navigation_map)
        terminal_ego_pose = _ego_pose(trajectory, len(frame_ids) - 1)
        frame_index = {
            frame_id: _write_frame(
                temporary,
                scene_name,
                index,
                frame_id,
                trajectory,
                terminal_ego_pose,
                cameras,
                scene_info,
                clone_images,
                num_future_trajectory_steps,
            )
            for index, frame_id in enumerate(frame_ids)
        }
        _write_scene_index(temporary, scene_name, frame_index)
        if visualize_ego_pose:
            _ego_pose_visualization(temporary, trajectory, frame_ids)
        if visualize_future_trajectory:
            _write_future_trajectory_visualizations(
                temporary, trajectory, frame_ids, num_future_trajectory_steps
            )
        temporary.rename(output_root / scene_name)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return scene_name


def _load_camera(
    camera_dir: Path,
    source_scene_id: str,
    episode: int,
    frame_count: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    camera_id = camera_dir.name
    try:
        rgb_summary = _load_json(camera_dir / "rgb_render_summary.json")
        depth_summary = _load_json(camera_dir / "depth_render_summary.json")
        _validate_summary(rgb_summary, profile, source_scene_id, camera_id, "rgb")
        _validate_summary(depth_summary, profile, source_scene_id, camera_id, "depth")
        width, height = _profile_resolution(profile)
        rgb_files = _frame_files(
            camera_dir / "observation.images.rgb", episode, "jpg", frame_count
        )
        depth_files = _frame_files(
            camera_dir / "observation.images.depth", episode, "png", frame_count
        )
        for path in rgb_files:
            _validate_image(path, width, height, depth=False)
        for path in depth_files:
            _validate_image(path, width, height, depth=True)
        mask_file = camera_dir / "valid_pixel_mask.png"
        _validate_mask(mask_file, width, height)
        return {
            "profile": profile,
            "calibration": _calibration(profile),
            "rgb_files": rgb_files,
            "depth_files": depth_files,
            "depth_scale": float(depth_summary["depth_scale"]),
            "mask_file": mask_file,
        }
    except Exception as error:
        raise ValueError(f"camera={camera_id}: {error}") from error


def _load_cameras(
    rendered: Path,
    source_scene_id: str,
    episode: int,
    frame_count: int,
    profiles: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    camera_dirs = [
        path
        for path in sorted(rendered.iterdir(), key=lambda path: path.name)
        if path.is_dir() and path.name in profiles
    ]
    if not camera_dirs:
        raise ValueError(
            "no rendered camera matches trajectory manifest camera_profiles"
        )
    return {
        camera_dir.name: _load_camera(
            camera_dir,
            source_scene_id,
            episode,
            frame_count,
            profiles[camera_dir.name],
        )
        for camera_dir in camera_dirs
    }


def _load_scene_inputs(
    scene_dir: Path,
) -> tuple[dict[str, Any], float, list[Any], dict[int, dict[str, Any]], dict[str, Any]]:
    source_scene_id = scene_dir.name
    manifest = _load_json(scene_dir / "trajectories" / "trajectory_manifest.json")
    candidates = _load_json(
        scene_dir / "optimized_trajectories" / "candidate_metadata.json"
    )
    validation = _load_json(
        scene_dir / "optimized_trajectories" / "validation_metadata.json"
    )
    for name, metadata in (
        ("manifest", manifest),
        ("candidate", candidates),
        ("validation", validation),
    ):
        if str(metadata.get("scene_id")) != source_scene_id:
            raise ValueError(f"{name} scene_id mismatch")
    profiles = manifest.get("camera_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("trajectory manifest has no camera_profiles")
    control_dt = float(candidates.get("effective_config", {}).get("control_dt_s"))
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("effective_config.control_dt_s must be positive")
    records, _ = _metadata_episode_records(candidates, "candidate", require_list=True)
    _, validated = _metadata_episode_records(validation, "validation")
    return profiles, control_dt, records, validated, manifest


def _validate_episode_record(
    record: dict[str, Any], episode: int, validated: dict[int, dict[str, Any]]
) -> tuple[str, float]:
    expected_name = f"episode_{episode:06d}.npz"
    if record.get("success") is not True or record.get("status") != "success":
        raise ValueError("candidate is not successful")
    if record.get("npz_filename") != expected_name:
        raise ValueError(f"npz_filename must be {expected_name}")
    t_output = float(record["T_output"])
    if not math.isfinite(t_output) or t_output < 0:
        raise ValueError("T_output must be finite and nonnegative")
    validation_record = validated.get(episode)
    if not validation_record or validation_record.get("validated") is not True:
        raise ValueError("independent validation did not pass")
    return expected_name, t_output


if __name__ == "__main__":
    sys.exit(main())
