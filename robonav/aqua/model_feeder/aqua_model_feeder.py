import torch
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, CameraDepthSet, EgoPoseSet

from robonav.registry import MODEL_FEEDERS
from robonav.common.util import rt2mat


__all__ = ["AquaModelFeeder"]


@MODEL_FEEDERS.register_module()
class AquaModelFeeder(BaseModelFeeder):
    def __init__(
        self,
        *args,
        debug=False,
        pe_downsample_factor=2,
        pe_range=(-5, -5, 5, 5),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.pe_downsample_factor = pe_downsample_factor
        self.pe_range = torch.tensor(pe_range)

    def process(self, frame_batch: list) -> dict | list:
        processed_batch: list[dict] = []
        for frame in frame_batch:
            frame_out = self._init_frame_out(frame)
            frame_out.update(**self._process_transformables(frame))
            self._create_camera_pe(frame_out)
            if self.debug:
                frame_out["transformables"] = frame["transformables"]
            processed_batch.append(frame_out)

        return processed_batch

    def _init_frame_out(self, frame):
        return {"index_info": frame["index_info"]}

    def _process_transformables(self, frame) -> dict:
        transformables = frame["transformables"]
        out = {}
        for _, trsfmb in transformables.items():
            if isinstance(trsfmb, CameraImageSet):
                cam_ids, camera_images = zip(*trsfmb.transformables.items())
                img_tensor = torch.vstack(
                    [t.tensor["img"].unsqueeze(0) for t in camera_images]
                )
                ego_mask = torch.vstack(
                    [t.tensor["ego_mask"].unsqueeze(0) for t in camera_images]
                )
                out["camera_images"] = img_tensor
                out["ego_masks"] = ego_mask
                out["camera_ids"] = cam_ids
                out["camera_types"] = [t.cam_type for t in camera_images]
                intrinsics = [torch.as_tensor(t.intrinsic) for t in camera_images]
                max_len = max(len(i) for i in intrinsics)
                out["intrinsic"] = torch.stack(
                    [
                        torch.nn.functional.pad(i, (0, max_len - len(i)))
                        for i in intrinsics
                    ]
                )
                out["extrinsic"] = torch.vstack(
                    [
                        torch.tensor(rt2mat(*t.extrinsic, as_homo=True))[None]
                        for t in camera_images
                    ]
                )
                continue
            if isinstance(trsfmb, CameraDepthSet):
                _, camera_images = zip(*trsfmb.transformables.items())
                img_tensor = torch.vstack(
                    [t.tensor["img"].unsqueeze(0) for t in camera_images]
                )
                out["camera_depths"] = img_tensor
                continue
            if isinstance(trsfmb, EgoPoseSet):
                out["ego_poses"] = (frame["transformables"].get("ego_poses"),)
                continue
        return out

    def _create_camera_pe(self, frame_out_dict):
        """Create position embedding for camera.

        For each PE-grid pixel of each camera, computes the body-frame unit ray
        [rx, ry, rz], the ground-plane intersection normalized by pe_range
        [qx, qy], and a validity mask. Output shape [N, 6, H_pe, W_pe].
        """
        camera_images = frame_out_dict["camera_images"]
        device = camera_images.device
        intr = frame_out_dict["intrinsic"].to(device=device, dtype=torch.float32)
        extr = frame_out_dict["extrinsic"].to(device=device, dtype=torch.float32)
        cam_types = frame_out_dict["camera_types"]
        _, _, img_h, img_w = camera_images.shape

        s = self.pe_downsample_factor
        h_pe = -(-img_h // s)
        w_pe = -(-img_w // s)

        fx = intr[:, 2] / s
        fy = intr[:, 3] / s
        cx = intr[:, 0] / s
        cy = intr[:, 1] / s

        v, u = torch.meshgrid(
            torch.arange(h_pe, device=device),
            torch.arange(w_pe, device=device),
            indexing="ij",
        )
        u = u.to(torch.float32)[None]  # [1, H_pe, W_pe]
        v = v.to(torch.float32)[None]

        x_d = (u - cx[:, None, None]) / fx[:, None, None]
        y_d = (v - cy[:, None, None]) / fy[:, None, None]

        is_fisheye = torch.tensor(
            [ct == "FisheyeCamera" for ct in cam_types], device=device
        )

        # Perspective: normalized pinhole ray [x, y, 1]
        ray_persp = torch.stack([x_d, y_d, torch.ones_like(x_d)], dim=-1)
        ray_persp = ray_persp / torch.linalg.vector_norm(
            ray_persp, dim=-1, keepdim=True
        )

        # Fisheye: invert OpenCV theta polynomial, then build a spherical unit ray
        dist = torch.zeros((x_d.shape[0], 4), device=device, dtype=x_d.dtype)
        if is_fisheye.any():
            dist[is_fisheye] = intr[is_fisheye][:, 4:8]
        k1 = dist[:, 0][:, None, None]
        k2 = dist[:, 1][:, None, None]
        k3 = dist[:, 2][:, None, None]
        k4 = dist[:, 3][:, None, None]

        theta_d = torch.sqrt(x_d**2 + y_d**2)
        theta = theta_d.clone()
        for _ in range(8):
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta4 * theta2
            theta8 = theta4**2
            f = (
                theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
                - theta_d
            )
            df = (
                1.0
                + 3.0 * k1 * theta2
                + 5.0 * k2 * theta4
                + 7.0 * k3 * theta6
                + 9.0 * k4 * theta8
            )
            theta = theta - f / df

        eps_center = 1e-8
        non_center = theta_d > eps_center
        radial_scale = torch.where(
            non_center, torch.sin(theta) / theta_d, torch.zeros_like(theta_d)
        )
        ray_fish = torch.stack(
            [
                x_d * radial_scale,
                y_d * radial_scale,
                torch.where(non_center, torch.cos(theta), torch.ones_like(theta)),
            ],
            dim=-1,
        )
        ray_fish = ray_fish / torch.linalg.vector_norm(ray_fish, dim=-1, keepdim=True)

        ray_c = torch.where(is_fisheye[:, None, None, None], ray_fish, ray_persp)

        # Camera frame -> body frame: rotate direction by R only, origin is t
        R = extr[:, :3, :3]
        t = extr[:, :3, 3]
        ray_b = torch.einsum("nij,nhwj->nhwi", R, ray_c)
        rx = ray_b[..., 0]
        ry = ray_b[..., 1]
        rz = ray_b[..., 2]

        # Intersect ray with ground plane z = 0
        eps_ground = 1e-6
        non_parallel = torch.abs(rz) > eps_ground
        safe_rz = torch.where(non_parallel, rz, torch.ones_like(rz))
        lam = (0.0 - t[:, None, None, 2]) / safe_rz
        valid = non_parallel & (lam > 0.0)

        qx = t[:, None, None, 0] + lam * rx
        qy = t[:, None, None, 1] + lam * ry

        # Clamp, normalize to [-1, 1], then zero out invalid coordinates
        x_min, y_min, x_max, y_max = self.pe_range.tolist()
        qx = torch.clamp(qx, x_min, x_max)
        qy = torch.clamp(qy, y_min, y_max)
        qx = 2.0 * (qx - x_min) / (x_max - x_min) - 1.0
        qy = 2.0 * (qy - y_min) / (y_max - y_min) - 1.0
        qx = torch.where(valid, qx, torch.zeros_like(qx))
        qy = torch.where(valid, qy, torch.zeros_like(qy))
        m_valid = valid.to(torch.float32)

        frame_out_dict["position_embedding"] = torch.stack(
            [rx, ry, rz, qx, qy, m_valid], dim=1
        )
