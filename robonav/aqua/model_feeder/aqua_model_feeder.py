import torch
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, CameraDepthSet

from robonav.registry import MODEL_FEEDERS


__all__ = ["AquaModelFeeder"]


@MODEL_FEEDERS.register_module()
class AquaModelFeeder(BaseModelFeeder):
    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

    def process(self, frame_batch: list) -> dict | list:
        processed_batch: list[dict] = []
        for frame in frame_batch:
            transformables = frame["transformables"]
            frame_out = {
                "index_info": frame["index_info"],
                "ego_poses": transformables.get("ego_poses"),
            }
            for name, trsfmb in transformables.items():
                if isinstance(trsfmb, CameraImageSet):
                    cam_ids, camera_images = zip(*trsfmb.transformables.items())
                    img_tensor = torch.vstack(
                        [t.tensor["img"].unsqueeze(0) for t in camera_images]
                    )
                    ego_mask = torch.vstack(
                        [t.tensor["ego_mask"].unsqueeze(0) for t in camera_images]
                    )
                    frame_out["camera_images"] = img_tensor
                    frame_out["ego_masks"] = ego_mask
                    frame_out["camera_ids"] = cam_ids
                    continue
                if isinstance(trsfmb, CameraDepthSet):
                    _, camera_images = zip(*trsfmb.transformables.items())
                    img_tensor = torch.vstack(
                        [t.tensor["img"].unsqueeze(0) for t in camera_images]
                    )
                    frame_out["camera_depths"] = img_tensor
                    continue

            if self.debug:
                frame_out["transformables"] = transformables

            processed_batch.append(frame_out)

        return processed_batch
