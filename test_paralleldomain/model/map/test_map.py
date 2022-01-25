import os

import pytest

from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.utilities.any_path import AnyPath
from test_paralleldomain.decoding.constants import DGP_V1_DATASET_PATH_ENV, UMD_FILE_PATH_ENV


@pytest.fixture
def umd_path() -> str:
    if UMD_FILE_PATH_ENV not in os.environ:
        pytest.skip()
    return os.environ[UMD_FILE_PATH_ENV]


@pytest.fixture
def umd_dataset_path() -> str:
    if DGP_V1_DATASET_PATH_ENV not in os.environ:
        pytest.skip()
    return os.environ[DGP_V1_DATASET_PATH_ENV]


def test_load_map(umd_path: str, umd_dataset_path: str):
    dataset = decode_dataset(
        dataset_path=umd_dataset_path,
        dataset_format="dgpv1",
        umd_file_paths={"scene_000000": AnyPath(umd_path)},
    )

    scene = dataset.get_scene(scene_name="scene_000000")
    map = scene.map
    assert map is not None
    assert map.lane_segments is not None
    assert len(map.lane_segments) > 0
    assert map.junctions is not None
    assert len(map.junctions) > 0
    assert map.road_segments is not None
    assert len(map.road_segments) > 0
    assert map.areas is not None
    assert len(map.areas) > 0
    segment = list(map.lane_segments.values())[0]
    assert len(map.get_lane_segment_predecessors_random_path(lane_segment=segment, steps=5)) > 0
    assert len(map.get_lane_segment_successors_random_path(lane_segment=segment, steps=5)) > 0
