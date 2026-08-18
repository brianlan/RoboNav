import numpy as np
import pytest
import torch
from mmengine.config import Config
from prefusion.registry import TENSOR_SMITHS, TRANSFORMABLE_LOADERS

from robonav.aqua.model_feeder import AquaModelFeeder
from robonav.aqua.tensor_smith import NavigationMap2DTensorSmith
from robonav.aqua.transformable import NavigationMap2D
from robonav.aqua.transformable_loader import NavigationMap2DLoader


def test_navigation_map_default_tensor_contract():
    map_2d = NavigationMap2D(
        "map",
        np.full((2, 2), 127, np.uint8),
        np.ones((2, 2), np.float32),
        np.zeros((2, 2), bool),
        np.eye(3),
    )
    tensors = NavigationMap2DTensorSmith()(map_2d)
    assert tensors["occupancy"].shape == (3, 120, 120)
    assert tensors["clearance"].shape == tensors["traversability"].shape == (
        1,
        120,
        120,
    )
    assert tensors["occupancy"].dtype == torch.float32
    assert tensors["clearance"].dtype == torch.float32
    assert tensors["traversability"].dtype == torch.bool


def test_navigation_map_sampling_values_orientation_and_padding():
    map_2d = NavigationMap2D(
        "map",
        np.array([[127, 255], [0, 127]], dtype=np.uint8),
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        np.array([[True, False], [False, True]]),
        np.eye(3),
    )
    tensors = NavigationMap2DTensorSmith(
        output_range=(-0.5, -0.5, 2.5, 2.5), resolution=1
    )(map_2d)
    expected_occupancy = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tensors["occupancy"][:, :2, :2], expected_occupancy)
    torch.testing.assert_close(
        tensors["clearance"][0, :2, :2],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )
    assert torch.equal(
        tensors["traversability"][0, :2, :2],
        torch.tensor([[True, False], [False, True]]),
    )
    assert torch.equal(tensors["occupancy"][:, 2, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert tensors["clearance"][0, 2, 2] == 0
    assert not tensors["traversability"][0, 2, 2]


def test_navigation_map_rejects_nondivisible_grid_and_bad_rasters():
    with pytest.raises(ValueError):
        NavigationMap2DTensorSmith(output_range=(0, 0, 1, 1), resolution=0.3)
    with pytest.raises(ValueError):
        NavigationMap2D(
            "map",
            np.zeros((2, 2), np.uint8),
            np.ones((2, 2), np.float32),
            np.ones((2, 2)),
            np.eye(3),
        )


def test_navigation_map_composes_planar_transforms_and_rejects_tilt():
    map_2d = NavigationMap2D(
        "map",
        np.full((2, 2), 127, np.uint8),
        np.ones((2, 2)),
        np.ones((2, 2), bool),
        np.eye(3),
    )
    originals = tuple(
        raster.copy()
        for raster in (map_2d.occupancy, map_2d.clearance, map_2d.traversability)
    )
    yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    flip = np.diag([-1, 1, 1.0])
    translate = np.eye(3)
    translate[:2, 2] = [-1, -2]
    map_2d.translate_3d([1, 2, 9]).flip_3d(flip).rotate_3d(yaw)
    np.testing.assert_allclose(map_2d.source_pixel_to_body, yaw @ flip @ translate)
    for raster, original in zip(
        (map_2d.occupancy, map_2d.clearance, map_2d.traversability), originals
    ):
        np.testing.assert_array_equal(raster, original)
        assert not raster.flags.writeable
    with pytest.raises(ValueError):
        map_2d.rotate_3d(np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]]))
    with pytest.raises(ValueError):
        map_2d.flip_3d(np.diag([1, 1, -1]))
    with pytest.raises(ValueError):
        map_2d.flip_3d(np.array([[1, 0, 0.1], [0, 1, 0], [0, 0, 1.]]))
    with pytest.raises(ValueError):
        map_2d.rotate_3d(np.diag([1, 1, -1]))
    with pytest.raises(ValueError):
        map_2d.rotate_3d(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1.]]))


def test_loader_resolves_relative_paths_and_composes_ego_affine(tmp_path):
    from PIL import Image

    (tmp_path / "scene" / "map").mkdir(parents=True)
    Image.fromarray(np.array([[127, 0], [255, 127]], dtype=np.uint8)).save(
        tmp_path / "scene/map/occupancy.png"
    )
    np.save(tmp_path / "scene/map/clearance.npy", np.ones((2, 2), np.float32))
    Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(
        tmp_path / "scene/map/traversability.png"
    )
    info = {
        "occupancy_path": "scene/map/occupancy.png",
        "clearance_path": "scene/map/clearance.npy",
        "traversability_path": "scene/map/traversability.png",
        "pixel_to_world": [[2, 0, 3], [0, 2, 4], [0, 0, 1]],
    }
    ego = {
        "rotation": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        "translation": np.array([5, 6, 0]),
    }
    loaded = NavigationMap2DLoader(tmp_path).load(
        "map",
        None,
        {"scene_info": {"navigation_map_2d": info}, "ego_pose": ego},
        None,
    )
    expected = np.array([[0, 2, -2], [-2, 0, 2], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(loaded.source_pixel_to_body, expected)
    assert loaded.clearance.dtype == np.float32
    assert loaded.traversability.dtype == np.bool_
    assert all(
        not raster.flags.writeable
        for raster in (loaded.occupancy, loaded.clearance, loaded.traversability)
    )

    Image.fromarray(np.array([[127, 0], [0, 255]], dtype=np.uint8)).save(
        tmp_path / "scene/map/traversability.png"
    )
    with pytest.raises(ValueError, match="only 0 and 255"):
        NavigationMap2DLoader(tmp_path).load(
            "map",
            None,
            {"scene_info": {"navigation_map_2d": info}, "ego_pose": ego},
            None,
        )


def test_feeder_and_config_wiring(tmp_path):
    map_2d = NavigationMap2D(
        "map",
        np.full((2, 2), 127, np.uint8),
        np.ones((2, 2), np.float32),
        np.ones((2, 2), bool),
        np.eye(3),
    )
    map_2d.tensor = {
        "occupancy": torch.ones(3, 2, 2),
        "clearance": torch.ones(1, 2, 2),
        "traversability": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    out = AquaModelFeeder()._process_transformables(
        {"transformables": {"navigation_map_2d": map_2d}}
    )
    assert set(out) == {"occupancy", "clearance", "traversability"}

    config = Config.fromfile("configs/kinogoal_dla_resnet18_overfit.py")
    wiring = config.transformables.navigation_map_2d
    loader = TRANSFORMABLE_LOADERS.build({**wiring.loader, "data_root": tmp_path})
    smith = TENSOR_SMITHS.build(wiring.tensor_smith)
    assert isinstance(loader, NavigationMap2DLoader)
    assert isinstance(smith, NavigationMap2DTensorSmith)
    assert smith.output_range == (-1, -3, 5, 3) and smith.resolution == 0.05
