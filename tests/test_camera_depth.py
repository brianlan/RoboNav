import numpy as np
import pytest
import torch
from prefusion.dataset.transform import CameraDepth, CameraDepthSet

from robonav.aqua.model_feeder import AquaModelFeeder
from robonav.aqua.tensor_smith import CameraDepthTensor


def _depth(cam_id, image, ego_mask, depth_mode="d"):
    return CameraDepth(
        name="camera_depths",
        cam_id=cam_id,
        cam_type="FisheyeCamera",
        img=np.asarray(image, dtype=np.float32),
        ego_mask=np.asarray(ego_mask, dtype=np.uint8),
        extrinsic=(np.eye(3), np.zeros(3)),
        intrinsic=[0, 0, 1, 1, 0, 0, 0, 0],
        depth_mode=depth_mode,
        tensor_smith=CameraDepthTensor(max_depth=5),
    )


def test_camera_depth_tensor_requires_euclidean_ray_distance():
    distance = _depth("cam0", [[1]], [[1]])
    distance.to_tensor()
    assert distance.tensor["img"].item() == pytest.approx(0.2)

    z_depth = _depth("cam0", [[1]], [[1]], depth_mode="z")
    with pytest.raises(ValueError, match="depth_mode='d'.*Euclidean ray distance"):
        z_depth.to_tensor()


def test_camera_depth_tensor_channel_cutoff_and_validity():
    depth = _depth(
        "cam0",
        [[0, -1, np.nan, np.inf, 2.5], [5, 6, 100, 1, 4]],
        [[1, 1, 1, 1, 1], [1, 1, 0, 1, 0]],
    )
    depth.to_tensor()

    image = depth.tensor["img"]
    valid = depth.tensor["valid_mask"]
    assert image.shape == valid.shape == (1, 2, 5)
    assert image.dtype == torch.float32
    assert valid.dtype == torch.bool
    torch.testing.assert_close(
        image,
        torch.tensor([[[0, 0, 0, 0, 0.5], [1, 1, 0, 0.2, 0]]]),
    )
    assert torch.equal(
        valid,
        torch.tensor(
            [[[False, False, False, False, True], [True, True, False, True, False]]]
        ),
    )


def test_camera_depth_tensor_accepts_depth_plane_with_optional_channel():
    image = np.array([[1, 6], [0, 2]], dtype=np.float32)
    ego_mask = [[1, 1], [1, 0]]
    depth_2d = _depth("cam0", image, ego_mask)
    depth_3d = _depth("cam0", image[..., None], ego_mask)

    depth_2d.to_tensor()
    depth_3d.to_tensor()

    assert depth_2d.tensor["img"].shape == (1, 2, 2)
    assert depth_2d.tensor["valid_mask"].shape == (1, 2, 2)
    assert torch.equal(depth_2d.tensor["img"], depth_3d.tensor["img"])
    assert torch.equal(depth_2d.tensor["valid_mask"], depth_3d.tensor["valid_mask"])

    multi_channel = _depth("cam0", np.ones((2, 2, 3)), ego_mask)
    with pytest.raises(ValueError, match="shape H,W or H,W,1"):
        multi_channel.to_tensor()


def test_model_feeder_preserves_camera_order_and_shapes():
    first = _depth("left", [[1, 2], [3, 4]], [[1, 1], [1, 1]])
    second = _depth("right", [[6, 0], [2, 3]], [[1, 1], [0, 1]])
    for depth in (first, second):
        depth.to_tensor()
    depth_set = CameraDepthSet("camera_depths", {"left": first, "right": second})
    frame = {"transformables": {"camera_depths": depth_set}}

    output = AquaModelFeeder()._process_transformables(frame)
    assert output["camera_depths"].shape == (2, 1, 2, 2)
    assert output["camera_depth_valid_masks"].shape == (2, 1, 2, 2)
    assert output["camera_depth_valid_masks"].dtype == torch.bool
    torch.testing.assert_close(output["camera_depths"][0], first.tensor["img"])
    torch.testing.assert_close(output["camera_depths"][1], second.tensor["img"])
    assert torch.equal(
        output["camera_depth_valid_masks"][1], second.tensor["valid_mask"]
    )
