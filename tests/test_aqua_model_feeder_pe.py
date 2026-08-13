import numpy as np
import torch
import cv2
from prefusion.dataset.transform import CameraImage, CameraImageSet

from robonav.aqua.model_feeder.aqua_model_feeder import AquaModelFeeder
from robonav.aqua.tensor_smith.camera_tensor_smith import CameraImageTensor


def forward_camera_extrinsic(forward, left, up):
    """4x4 camera(right/down/forward) -> body(forward/left/up), camera facing +body-x."""
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [forward, left, up]
    return T


def run_pe(n_cams, h, w, cam_types, intrinsics, extrinsics, **feeder_kwargs):
    feeder = AquaModelFeeder(**feeder_kwargs)
    frame = {
        "camera_images": torch.randn(n_cams, 3, h, w),
        "camera_types": cam_types,
        "intrinsic": torch.tensor(intrinsics, dtype=torch.float32),
        "extrinsic": torch.tensor(extrinsics, dtype=torch.float32),
    }
    feeder._create_camera_pe(frame)
    return frame["position_embedding"]


def make_camera_image(cam_id, cam_type, intrinsic, h, w):
    T = forward_camera_extrinsic(0, 0, 1)
    return CameraImage(
        name="camera_images",
        cam_id=cam_id,
        cam_type=cam_type,
        img=np.zeros((h, w, 3), dtype=np.uint8),
        ego_mask=np.zeros((h, w), dtype=np.uint8),
        extrinsic=(T[:3, :3], T[:3, 3]),
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        tensor_smith=CameraImageTensor(),
    )


def test_process_mixed_camera_types():
    h, w = 64, 96
    persp = make_camera_image("cam0", "PerspectiveCamera", [16, 24, 20, 20], h, w)
    fish = make_camera_image(
        "cam1", "FisheyeCamera", [16, 24, 20, 20, 0.05, -0.01, 0.002, 0.001], h, w
    )
    cam_set = CameraImageSet("camera_images", {"cam0": persp, "cam1": fish})
    for cam in cam_set.transformables.values():
        cam.to_tensor()
    frame = {"index_info": {}, "transformables": {"camera_images": cam_set}}
    out = AquaModelFeeder().process([frame])[0]

    assert out["camera_types"] == ["PerspectiveCamera", "FisheyeCamera"]
    # mixed 4/8-value intrinsics are stacked into a rectangular tensor,
    # missing perspective distortion coefficients zero-padded
    assert out["intrinsic"].shape == (2, 8)
    assert torch.all(out["intrinsic"][0, 4:] == 0)
    k = torch.tensor([0.05, -0.01, 0.002, 0.001], dtype=out["intrinsic"].dtype)
    assert torch.all(out["intrinsic"][1, 4:] == k)

    pe = out["position_embedding"]
    assert pe.shape == (2, 6, 32, 48)
    assert pe.dtype == torch.float32
    m = pe[:, 5]
    assert torch.all((m == 0) | (m == 1))
    assert torch.all(pe[:, 3] >= -1) and torch.all(pe[:, 3] <= 1)
    assert torch.all(pe[:, 4] >= -1) and torch.all(pe[:, 4] <= 1)


def test_float64_intrinsics_extrinsics_produce_float32():
    h, w = 64, 96
    feeder = AquaModelFeeder()
    camera_images = torch.randn(1, 3, h, w)
    frame = {
        "camera_images": camera_images,
        "camera_types": ["FisheyeCamera"],
        "intrinsic": torch.tensor(
            [[16, 24, 20, 20, 0.05, -0.01, 0.002, 0.001]], dtype=torch.float64
        ),
        "extrinsic": torch.tensor(
            [forward_camera_extrinsic(0.2, 0, 0.55)], dtype=torch.float64
        ),
    }
    feeder._create_camera_pe(frame)
    pe = frame["position_embedding"]
    assert pe.shape == (1, 6, 32, 48)
    assert pe.dtype == torch.float32
    assert pe.device == camera_images.device


def test_perspective_geometry_and_axis_rotation():
    h, w = 64, 64
    s = 2
    cx, cy, fx, fy = 16, 16, 8, 8
    pe = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [forward_camera_extrinsic(0, 0, 1)],
    )[0]  # [6, 32, 32]

    # optical axis (PE grid pixel cx/s, cy/s) -> body +x
    v0, u0 = cy // s, cx // s
    rx, ry, rz = pe[0, v0, u0], pe[1, v0, u0], pe[2, v0, u0]
    assert abs(rx - 1) < 1e-5 and abs(ry) < 1e-5 and abs(rz) < 1e-5

    # image right -> body left (ry < 0), image left -> ry > 0
    assert pe[1, v0, u0 + 4] < 0
    assert pe[1, v0, u0 - 4] > 0

    # image down -> rz < 0, image up -> rz > 0
    assert pe[2, v0 + 4, u0] < 0
    assert pe[2, v0 - 4, u0] > 0

    # hand-computed ground intersection: PE pixel (cx/s, (cy+fy)/s), ray 1/sqrt2 [1,0,-1]
    # camera at (0,0,1): lam = sqrt(2), q = (1, 0); pe_range (-5,-5,5,5) -> qhat (0.2, 0)
    v_g, u_g = (cy + fy) // s, cx // s
    assert abs(pe[0, v_g, u_g] - 1 / np.sqrt(2)) < 1e-5
    assert abs(pe[1, v_g, u_g]) < 1e-5
    assert abs(pe[2, v_g, u_g] + 1 / np.sqrt(2)) < 1e-5
    assert abs(pe[3, v_g, u_g] - 0.2) < 1e-5
    assert abs(pe[4, v_g, u_g]) < 1e-5
    assert pe[5, v_g, u_g] == 1


def test_fisheye_matches_cv2_oracle():
    h, w = 64, 64
    s = 2
    cx, cy, fx, fy = 32, 32, 50, 50
    k = [0.05, -0.01, 0.002, 0.001]
    pe = run_pe(
        1,
        h,
        w,
        ["FisheyeCamera"],
        [[cx, cy, fx, fy, *k]],
        [np.eye(4)],
    )[0]  # identity extrinsic -> body ray == camera ray

    ray = pe[:3].permute(1, 2, 0).reshape(-1, 3).numpy()

    v, u = torch.meshgrid(torch.arange(h // s), torch.arange(w // s), indexing="ij")
    K = np.array(
        [[fx / s, 0, cx / s], [0, fy / s, cy / s], [0, 0, 1]], dtype=np.float64
    )
    D = np.array([k], dtype=np.float64)
    pts = np.stack([u.numpy().ravel(), v.numpy().ravel()], axis=-1).astype(np.float64)[
        :, None, :
    ]
    out = cv2.fisheye.undistortPoints(pts, K, D, None, None).reshape(-1, 2)
    oracle = np.concatenate([out, np.ones((len(out), 1))], axis=-1)
    oracle = oracle / np.linalg.norm(oracle, axis=-1)[:, None]

    assert np.abs(ray - oracle).max() < 1e-4


def test_unit_ray_norms():
    h, w = 64, 96
    n_cams = 2
    cam_types = ["PerspectiveCamera", "FisheyeCamera"]
    intrinsics = [
        [16, 24, 20, 20, 0, 0, 0, 0],
        [16, 24, 20, 20, 0.05, -0.01, 0.002, 0.001],
    ]
    extrinsics = [forward_camera_extrinsic(0.2, 0, 0.55) for _ in range(n_cams)]
    pe = run_pe(n_cams, h, w, cam_types, intrinsics, extrinsics)

    norms = torch.linalg.vector_norm(pe[:, :3], dim=1)
    assert torch.all((norms - 1).abs() < 1e-4)


def test_ground_valid_invalid_and_exact_zero_q():
    h, w = 64, 64
    s = 2
    cx, cy, fx, fy = 16, 16, 8, 8
    pe = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [forward_camera_extrinsic(0, 0, 1)],
    )[0]

    v0 = cy // s

    # horizon row (v == cy/s, rz == 0): invalid, q exactly zero
    horizon = pe[:, v0]
    assert torch.all(horizon[5] == 0)
    assert torch.all(horizon[3] == 0)
    assert torch.all(horizon[4] == 0)

    # upper half points up: invalid with exactly-zero q
    assert torch.all(pe[5, :v0] == 0)
    assert torch.all(pe[3, :v0] == 0)
    assert torch.all(pe[4, :v0] == 0)

    # lower half: valid, nonzero intersection
    assert torch.all(pe[5, v0 + 1 :] == 1)
    assert torch.all(pe[3, v0 + 1 :] != 0)

    # invalid -> q exactly zero holds everywhere
    invalid = pe[5] == 0
    assert torch.all(pe[3][invalid] == 0)
    assert torch.all(pe[4][invalid] == 0)


def test_pe_range_clamp_and_normalization():
    h, w = 64, 96
    cx, cy, fx, fy = 20, 20, 10, 10
    pe_range = (-1, -1, 1, 1)
    pe = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [forward_camera_extrinsic(0, 0, 0.5)],
        pe_range=pe_range,
    )[0]

    # PE grid pixels at k = 0.4 (v = cy/s + 0.4*fy/s) and k = 1.0
    v_lo = int(cy / 2 + 0.4 * fy / 2)
    v_hi = int(cy / 2 + 1.0 * fy / 2)
    u0 = cx // 2

    # qx = 1.25 -> clamped to 1 -> qhat_x = 1, still valid
    assert abs(pe[3, v_lo, u0] - 1.0) < 1e-5
    assert pe[5, v_lo, u0] == 1
    # qx = 0.5 -> qhat_x = 0.5
    assert abs(pe[3, v_hi, u0] - 0.5) < 1e-5
    assert pe[5, v_hi, u0] == 1
    # everything stays within [-1, 1]
    assert torch.all(pe[3] >= -1) and torch.all(pe[3] <= 1)


def test_downsample_intrinsic_scaling():
    h, w = 64, 64
    cx, cy, fx, fy = 32, 32, 20, 20
    ext = forward_camera_extrinsic(0.2, 0, 0.55)
    pe1 = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [ext],
        pe_downsample_factor=1,
    )[0]
    pe2 = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [ext],
        pe_downsample_factor=2,
    )[0]

    v, u = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    torch.testing.assert_close(pe2[:, v, u], pe1[:, 2 * v, 2 * u], atol=1e-5, rtol=0)


def test_odd_image_ceil_output():
    h, w = 33, 49
    cx, cy, fx, fy = 16, 16, 8, 8
    pe = run_pe(
        1,
        h,
        w,
        ["PerspectiveCamera"],
        [[cx, cy, fx, fy]],
        [forward_camera_extrinsic(0, 0, 1)],
    )
    assert pe.shape == (1, 6, 17, 25)

    pe = run_pe(
        1,
        65,
        63,
        ["PerspectiveCamera"],
        [[16, 16, 8, 8]],
        [forward_camera_extrinsic(0, 0, 1)],
    )
    assert pe.shape == (1, 6, 33, 32)
