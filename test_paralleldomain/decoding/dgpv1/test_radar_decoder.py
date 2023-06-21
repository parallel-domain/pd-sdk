import pytest
from paralleldomain import Dataset, Scene
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.model.frame import Frame
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.model.sensor import RadarSensorFrame
from typing import List
from test_paralleldomain.decoding.constants import RADAR_DATASET_V0_PATH_ENV
import os


@pytest.fixture
def radar_dataset_path() -> str:
    if RADAR_DATASET_V0_PATH_ENV in os.environ:
        return os.environ[RADAR_DATASET_V0_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def radar_dataset(radar_dataset_path: AnyPath) -> Dataset:
    decoder = DGPDatasetDecoder(dataset_path=radar_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def radar_scene_list(radar_dataset: Dataset) -> List:
    scene_names = radar_dataset.scene_names
    return scene_names


@pytest.fixture()
def radar_scene(radar_dataset: Dataset, scenes_list: List) -> Scene:
    scene = radar_dataset.get_scene(scene_name=scenes_list[0])
    return scene


@pytest.fixture()
def scenes_list(radar_dataset: Dataset) -> List:
    scene_list = radar_dataset.scene_names
    return scene_list


@pytest.fixture()
def radar_frame(radar_scene: Scene) -> RadarSensorFrame:
    frame_ids = radar_scene.frame_ids
    frame: Frame = radar_scene.get_frame(frame_id=frame_ids[0])
    radar_name = frame.radar_names[0]
    radar_frame: RadarSensorFrame = frame.get_radar(radar_name=radar_name)
    return radar_frame


def test_load_scene(radar_scene_list: List, radar_dataset: Dataset):
    for scene_name in radar_scene_list:
        scene = radar_dataset.get_scene(scene_name=scene_name)
        assert scene is not None


def test_decode_pointcloud(radar_frame: RadarSensorFrame):
    radar_point_cloud = radar_frame.radar_point_cloud.xyz
    assert radar_point_cloud is not None
    assert len(radar_point_cloud) > 0


def test_decode_frame_header(radar_frame: RadarSensorFrame):
    header = radar_frame.header
    assert header.timestamp is not None
    assert header.timestamp > 0
    assert header.max_non_ambiguous_doppler is not None
    assert header.max_non_ambiguous_doppler > 0
