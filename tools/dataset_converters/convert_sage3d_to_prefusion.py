"""Convert validated Sage3D episodes to independent prefusion scenes."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene-root", type=Path, required=True)
    parser.add_argument("--output-scene-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--clone-camera-images", action="store_true", default=False)
    return parser.parse_args(argv)


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
    coefficients = profile.get("fisheye_coefficients", profile.get("distortion_coefficients"))
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
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
        if not math.isfinite(theta_d) or theta_d <= 0:
            raise ValueError("opencv_fisheye angular polynomial must be finite and positive")
        focal = (width / 2.0) / theta_d
    if isinstance(focal, (list, tuple)):
        fx, fy = map(float, focal)
    elif focal is not None:
        fx = fy = float(focal)
    else:
        raise ValueError("camera profile lacks focal length")
    if not np.isfinite([fx, fy]).all() or fx <= 0 or fy <= 0 or not np.isclose(fx, fy):
        raise ValueError("camera profile must have finite positive square-pixel focal length")
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
    translation = np.asarray(extrinsic.get("translation_body_m"), dtype=np.float64)
    rpy = np.asarray(extrinsic.get("rotation_rpy_deg"), dtype=np.float64)
    if translation.shape != (3,) or rpy.shape != (3,) or not np.isfinite([translation, rpy]).all():
        raise ValueError("camera extrinsic must contain finite xyz translation and RPY")
    rotation = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    return rotation, translation


def _calibration(profile: dict[str, Any]) -> dict[str, Any]:
    rotation, translation = _profile_extrinsic(profile)
    return {
        "camera_type": "PerspectiveCamera" if _profile_model(profile) == "pinhole" else "FisheyeCamera",
        "intrinsic": np.asarray(_profile_intrinsic(profile), dtype=np.float64),
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
    _same(summary.get("camera_model"), _profile_model(profile), f"{mode} camera model")
    _same(summary.get("resolution"), list(_profile_resolution(profile)), f"{mode} resolution")
    intrinsic = _profile_intrinsic(profile)
    _same(summary.get("principal_point"), intrinsic[:2], f"{mode} principal point")
    _same(summary.get("focal_length_pixels"), intrinsic[2], f"{mode} focal length")
    summary_extrinsic = summary.get("camera_extrinsic")
    profile_extrinsic = profile.get("extrinsic")
    if not isinstance(summary_extrinsic, dict):
        raise ValueError(f"{mode} summary lacks camera_extrinsic")
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
    if _profile_model(profile) == "opencv_fisheye":
        _same(summary.get("fisheye_coefficients"), intrinsic[4:], f"{mode} fisheye coefficients")
        radius = float(summary.get("forward_mask_radius_pixels", 0))
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"{mode} summary has invalid forward_mask_radius_pixels")
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


def _frame_files(directory: Path, episode: int, suffix: str, frame_count: int) -> list[Path]:
    pattern = re.compile(rf"^episode_{episode:06d}_(\d+)\.{re.escape(suffix)}$", re.IGNORECASE)
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
        raise ValueError(f"{suffix} frame indices mismatch; missing={missing} extra={extra}")
    return [indexed[index] for index in range(frame_count)]


def _validate_image(path: Path, width: int, height: int, *, depth: bool) -> None:
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError(f"image resolution {image.size} != {(width, height)}")
        array = np.asarray(image)
    if depth:
        if array.shape != (height, width) or array.dtype != np.uint16:
            raise ValueError(f"depth must be 2D uint16, got shape={array.shape} dtype={array.dtype}")
    elif array.shape != (height, width, 3) or array.dtype != np.uint8:
        raise ValueError(f"RGB must be HxWx3 uint8, got shape={array.shape} dtype={array.dtype}")


def _trajectory(npz_path: Path, control_dt: float, t_output: float) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        required = ("time_s", "pose_world", "velocity_world_mps", "yaw_rate_radps")
        arrays = {key: np.asarray(archive[key]) for key in required}
    time_s = arrays["time_s"]
    if time_s.ndim != 1 or len(time_s) == 0 or not np.isfinite(time_s).all():
        raise ValueError("time_s must be a non-empty finite 1D array")
    frame_count = len(time_s)
    if not np.isclose(time_s[0], 0.0, atol=1e-8):
        raise ValueError("time_s must start at zero")
    if frame_count > 1 and not np.allclose(np.diff(time_s), control_dt, rtol=1e-6, atol=1e-8):
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
    ids = [str(int(round(initial_ms + index * step_ms))) for index in range(frame_count)]
    integers = list(map(int, ids))
    if len(set(ids)) != frame_count or any(b <= a for a, b in zip(integers, integers[1:])):
        raise ValueError("control_dt_s does not produce unique strictly increasing millisecond frame IDs")
    return ids


def _write_mask(path: Path, profile: dict[str, Any], summary: dict[str, Any]) -> None:
    width, height = _profile_resolution(profile)
    mask = np.full((height, width), 255, dtype=np.uint8)
    if _profile_model(profile) == "opencv_fisheye":
        radius = float(summary["forward_mask_radius_pixels"])
        cx, cy = _profile_intrinsic(profile)[:2]
        yy, xx = np.ogrid[:height, :width]
        mask = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2, 255, 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(path)


def _copy_or_link(source: Path, destination: Path, clone: bool) -> None:
    if clone:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def _write_episode(
    source_scene_id: str,
    episode: int,
    output_root: Path,
    trajectory: dict[str, np.ndarray],
    frame_ids: list[str],
    cameras: dict[str, dict[str, Any]],
    clone_images: bool,
) -> str:
    scene_name = f"sage3d-{source_scene_id}-{episode:06d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".{scene_name}.", dir=output_root))
    try:
        frame_dir = temporary / "frame_info_pkl"
        frame_dir.mkdir()
        for camera_id, camera in cameras.items():
            (temporary / "samples" / camera_id).mkdir(parents=True)
            (temporary / "camera_image_depth" / camera_id).mkdir(parents=True)
        (temporary / "self_mask").mkdir()

        calibrations = {camera_id: camera["calibration"] for camera_id, camera in cameras.items()}
        masks = {}
        for camera_id, camera in cameras.items():
            mask_relative = Path(scene_name) / "self_mask" / f"{camera_id}.png"
            masks[camera_id] = str(mask_relative)
            _write_mask(temporary / "self_mask" / f"{camera_id}.png", camera["profile"], camera["rgb_summary"])

        frame_index = {}
        for index, frame_id in enumerate(frame_ids):
            camera_images = {}
            depth_images = {}
            for camera_id, camera in cameras.items():
                rgb_name = f"{frame_id}.jpg"
                depth_name = f"{frame_id}.npz"
                rgb_relative = Path(scene_name) / "samples" / camera_id / rgb_name
                depth_relative = Path(scene_name) / "camera_image_depth" / camera_id / depth_name
                _copy_or_link(camera["rgb_files"][index], temporary / "samples" / camera_id / rgb_name, clone_images)
                with Image.open(camera["depth_files"][index]) as image:
                    depth = np.asarray(image, dtype=np.uint16).astype(np.float32) / camera["depth_scale"]
                np.savez_compressed(temporary / "camera_image_depth" / camera_id / depth_name, depth=depth)
                camera_images[camera_id] = {
                    "path": str(rgb_relative),
                    "calibration": camera["calibration"],
                }
                depth_images[camera_id] = str(depth_relative)

            x, y, yaw = trajectory["pose_world"][index]
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            translation = np.asarray([x, y, 0.0], dtype=np.float64)
            velocity_world = np.asarray([*trajectory["velocity_world_mps"][index], 0.0], dtype=np.float64)
            angular_world = np.asarray([0.0, 0.0, trajectory["yaw_rate_radps"][index]], dtype=np.float64)
            ego_pose = {
                "rotation": rotation,
                "translation": translation,
                "linear_velocity": rotation.T @ velocity_world,
                "angular_velocity": rotation.T @ angular_world,
            }
            if not all(np.isfinite(value).all() for value in ego_pose.values()):
                raise ValueError(f"frame {index} ego pose contains non-finite values")
            frame_data = {
                "camera_image": camera_images,
                "camera_image_depth": depth_images,
                "ego_pose": ego_pose,
                "scene_info": {"camera_mask": masks, "calibration": calibrations},
            }
            frame_path = frame_dir / f"{frame_id}.pkl"
            with frame_path.open("wb") as stream:
                pickle.dump(frame_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            frame_index[frame_id] = str(Path(scene_name) / "frame_info_pkl" / frame_path.name)

        info = {scene_name: {"scene_info": {}, "frame_info": frame_index}}
        with (temporary / "info.pkl").open("wb") as stream:
            pickle.dump(info, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.rename(output_root / scene_name)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return scene_name


def _camera_data(
    rendered: Path,
    source_scene_id: str,
    episode: int,
    frame_count: int,
    profiles: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    actual = [path for path in sorted(rendered.iterdir(), key=lambda path: path.name) if path.is_dir() and path.name in profiles]
    if not actual:
        raise ValueError("no rendered camera matches trajectory manifest camera_profiles")
    cameras = {}
    for camera_dir in actual:
        camera_id = camera_dir.name
        try:
            profile = profiles[camera_id]
            rgb_summary = _load_json(camera_dir / "rgb_render_summary.json")
            depth_summary = _load_json(camera_dir / "depth_render_summary.json")
            _validate_summary(rgb_summary, profile, source_scene_id, camera_id, "rgb")
            _validate_summary(depth_summary, profile, source_scene_id, camera_id, "depth")
            width, height = _profile_resolution(profile)
            rgb_files = _frame_files(camera_dir / "observation.images.rgb", episode, "jpg", frame_count)
            depth_files = _frame_files(camera_dir / "observation.images.depth", episode, "png", frame_count)
            for path in rgb_files:
                _validate_image(path, width, height, depth=False)
            for path in depth_files:
                _validate_image(path, width, height, depth=True)
            cameras[camera_id] = {
                "profile": profile,
                "calibration": _calibration(profile),
                "rgb_summary": rgb_summary,
                "rgb_files": rgb_files,
                "depth_files": depth_files,
                "depth_scale": float(depth_summary["depth_scale"]),
            }
        except Exception as error:
            raise ValueError(f"camera={camera_id}: {error}") from error
    return cameras


def _process_scene(
    scene_dir: Path,
    output_root: Path,
    clone_images: bool,
    reasons: Counter[str],
) -> tuple[int, int, int]:
    source_scene_id = scene_dir.name
    try:
        manifest = _load_json(scene_dir / "trajectories" / "trajectory_manifest.json")
        candidates = _load_json(scene_dir / "optimized_trajectories" / "candidate_metadata.json")
        validation = _load_json(scene_dir / "optimized_trajectories" / "validation_metadata.json")
        for name, metadata in (("manifest", manifest), ("candidate", candidates), ("validation", validation)):
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
    except Exception as error:
        _warning(source_scene_id, None, None, str(error))
        reasons["invalid source scene"] += 1
        return 0, 1, 0

    succeeded = skipped = existing = 0
    for record in records:
        episode: int | None = None
        try:
            episode = int(record["episode_index"])
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
            npz_path = scene_dir / "optimized_trajectories" / expected_name
            target = output_root / f"sage3d-{source_scene_id}-{episode:06d}"
            if target.exists():
                _warning(source_scene_id, episode, None, "target scene already exists")
                reasons["target exists"] += 1
                skipped += 1
                existing += 1
                continue
            trajectory = _trajectory(npz_path, control_dt, t_output)
            frame_ids = _frame_ids(npz_path, len(trajectory["time_s"]), control_dt)
            cameras = _camera_data(
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
            )
            succeeded += 1
        except Exception as error:
            message = str(error)
            camera_match = re.search(r"camera=([^:]+):", message)
            _warning(source_scene_id, episode, camera_match.group(1) if camera_match else None, message)
            reasons["invalid episode"] += 1
            skipped += 1
    return succeeded, skipped, existing


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    input_root = args.input_scene_root
    output_root = args.output_scene_root
    if not input_root.is_dir():
        logger.error("input scene root is not a directory: {}", input_root)
        return 2
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        if args.scene_ids is None:
            scene_dirs = sorted((path for path in input_root.iterdir() if path.is_dir()), key=lambda path: path.name)
        else:
            scene_dirs = [input_root / scene_id for scene_id in args.scene_ids]
        if not scene_dirs:
            logger.error("no source scenes selected")
            return 2
        reasons: Counter[str] = Counter()
        succeeded = skipped = existing = 0
        for scene_dir in tqdm(scene_dirs):
            if not scene_dir.is_dir():
                _warning(scene_dir.name, None, None, "source scene directory is missing")
                reasons["missing source scene"] += 1
                skipped += 1
                continue
            scene_succeeded, scene_skipped, scene_existing = _process_scene(
                scene_dir, output_root, args.clone_camera_images, reasons
            )
            succeeded += scene_succeeded
            skipped += scene_skipped
            existing += scene_existing
    except Exception as error:
        logger.error("fatal conversion error: {}", error)
        return 2

    reason_summary = ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())) or "none"
    logger.info(
        "summary source_scenes={} successful_episodes={} skipped={} reasons={}",
        len(scene_dirs),
        succeeded,
        skipped,
        reason_summary,
    )
    return 0 if succeeded > 0 or existing > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
