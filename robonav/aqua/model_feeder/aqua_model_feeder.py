import torch
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, CameraDepthSet, EgoPoseSet

from robonav.registry import MODEL_FEEDERS
from robonav.common.util import rt2mat


__all__ = ["AquaModelFeeder"]


@MODEL_FEEDERS.register_module()
class AquaModelFeeder(BaseModelFeeder):
    def __init__(self, *args, debug=False, pe_downsample_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.pe_downsample_factor = pe_downsample_factor

    def process(self, frame_batch: list) -> dict | list:
        processed_batch: list[dict] = []
        for frame in frame_batch:
            frame_out = self._init_frame_out(frame)
            frame_out.update(**self._process_transformables(frame))
            frame_out.update(**self._create_camera_pe(frame_out))
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
                out["intrinsic"] = torch.vstack(
                    [torch.tensor(t.intrinsic)[None] for t in camera_images]
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
        """Create position embedding for camera"""
        intr, extr = frame_out_dict["intrinsic"], frame_out_dict["extrinsic"]
        a = 10
