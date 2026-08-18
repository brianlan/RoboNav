import numpy as np
import torch

from tools.visualize_navigation_sequence import (
    decode_occupancy,
    front_up_raster,
    map_extent,
    map_point,
    map_vector,
    render_frame,
)


def test_front_up_mapping_and_ego_axes():
    np.testing.assert_array_equal(front_up_raster(np.array([[1, 2], [3, 4]])), [[2, 1], [4, 3]])
    assert map_extent((-1, -2, 5, 3)) == (-3, 2, -1, 5)
    assert map_point((2, 3)) == (-3.0, 2.0)
    assert map_vector((4, 0)) == (0.0, 4.0)
    assert map_vector((0, 4)) == (-4.0, 0.0)


def test_occupancy_decoding_and_three_file_contract(tmp_path):
    occupancy = torch.tensor([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 1]]])
    frame = {
        "occupancy": occupancy,
        "clearance": torch.ones(1, 2, 2),
        "traversability": torch.ones(1, 2, 2, dtype=torch.bool),
        "transformables": {},
    }
    np.testing.assert_array_equal(decode_occupancy(occupancy), [[0, 1], [2, 2]])
    render_frame(frame, tmp_path, 0, "frame-0", (-1, -1, 1, 1), 1.0)
    for kind in ("occupancy", "clearance", "traversability"):
        assert sorted(path.name for path in (tmp_path / kind).glob("*.png")) == [
            f"frame-0000-frame-0-{kind}.png",
        ]
    assert not list(tmp_path.glob("*.png"))
