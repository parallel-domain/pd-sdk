import os
from typing import List

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.flying_things.decoder import FlyingThingsDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from test_paralleldomain.decoding.constants import FLYING_THINGS_DATASET_PATH_ENV


@pytest.fixture
def flying_things_dataset_path() -> str:
    if FLYING_THINGS_DATASET_PATH_ENV in os.environ:
        return os.environ[FLYING_THINGS_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def flying_things_train_dataset(flying_things_dataset_path: str) -> Dataset:
    decoder = FlyingThingsDatasetDecoder(dataset_path=flying_things_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def flying_things_dataset_train_scenes(flying_things_train_dataset: Dataset) -> List:
    scene_names = flying_things_train_dataset.scene_names
    return scene_names


@pytest.fixture()
def flying_things_dataset_train_scene(
    flying_things_train_dataset: Dataset, flying_things_dataset_train_scenes: List
) -> Scene:
    scene = flying_things_train_dataset.get_scene(scene_name=flying_things_dataset_train_scenes[0])
    return scene


def test_can_load_scene(flying_things_dataset_train_scenes: List, flying_things_train_dataset: Dataset):
    for scene_name in flying_things_dataset_train_scenes:
        scene = flying_things_train_dataset.get_scene(scene_name=scene_name)
        assert scene is not None


def test_knows_all_frames(flying_things_dataset_train_scenes: List, flying_things_train_dataset: Dataset):
    for scene_name in flying_things_dataset_train_scenes:
        scene = flying_things_train_dataset.get_scene(scene_name=scene_name)
        assert 6 <= len(scene.frame_ids) <= 15  # FlyingThings3D: 6–15 frames for every scene


def test_decode_train_scene_names(flying_things_train_dataset: Dataset):
    # todo: find out proper number of frames for each split
    num_scenes = None
    assert len(flying_things_train_dataset.scene_names) == num_scenes
    assert len(flying_things_train_dataset.unordered_scene_names) == num_scenes


def test_decode_camera_image(flying_things_dataset_train_scene: Scene):
    camera_frame = next(flying_things_dataset_train_scene.camera_frames)
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    image = camera_frame.image
    assert image is not None
    rgb = image.rgb
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[0] == 540
    assert rgb.shape[1] == 960
    assert rgb.shape[2] == 3
    assert rgb.shape[0] == image.height
    assert rgb.shape[1] == image.width
    assert rgb.shape[2] == image.channels


def test_decode_optical_flow(flying_things_dataset_train_scene: Scene):
    camera_frame = flying_things_dataset_train_scene.get_frame(
        frame_id=flying_things_dataset_train_scene.frame_ids[0]
    ).get_sensor(sensor_name="left")
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    flow = camera_frame.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)
    assert flow is not None
    vectors = flow.vectors
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 540
    assert vectors.shape[1] == 960
    assert vectors.shape[2] == 2
    assert np.all(np.logical_and(flow.vectors.max() < 960, flow.vectors.min() >= -960))
