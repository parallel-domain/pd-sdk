import os
from typing import List

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from test_paralleldomain.decoding.constants import DGP_DATASET_PATH_ENV


@pytest.fixture
def dgp_dataset_path() -> str:
    if DGP_DATASET_PATH_ENV in os.environ:
        return os.environ[DGP_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def dgp_train_dataset(dgp_dataset_path: str) -> Dataset:
    decoder = DGPDatasetDecoder(dataset_path=dgp_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def dgp_dataset_train_scenes(dgp_train_dataset: Dataset) -> List:
    scene_names = dgp_train_dataset.scene_names
    return scene_names


@pytest.fixture()
def dgp_dataset_train_scene(dgp_train_dataset: Dataset, dgp_dataset_train_scenes: List) -> Scene:
    scene = dgp_train_dataset.get_scene(scene_name=dgp_dataset_train_scenes[0])
    return scene


def test_can_load_scene(dgp_dataset_train_scenes: List, dgp_train_dataset: Dataset):
    for scene_name in dgp_dataset_train_scenes:
        scene = dgp_train_dataset.get_scene(scene_name=scene_name)
        assert scene is not None


def test_knows_all_frames(dgp_dataset_train_scenes: List, dgp_train_dataset: Dataset):
    for scene_name in dgp_dataset_train_scenes:
        scene = dgp_train_dataset.get_scene(scene_name=scene_name)
        assert len(scene.frame_ids) > 0


def test_decode_train_scene_names(dgp_train_dataset: Dataset):
    assert len(dgp_train_dataset.scene_names) == 5
    assert len(dgp_train_dataset.unordered_scene_names) == 5


def test_decode_camera_image(dgp_dataset_train_scene: Scene):
    camera_frame = next(dgp_dataset_train_scene.camera_frames)
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    image = camera_frame.image
    assert image is not None
    rgb = image.rgb
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[0] == 1080
    assert rgb.shape[1] == 1920
    assert rgb.shape[2] == 3
    assert rgb.shape[0] == image.height
    assert rgb.shape[1] == image.width


def test_decode_camera_semseg_2d(dgp_dataset_train_scene: Scene):
    camera_frame = next(dgp_dataset_train_scene.camera_frames)
    semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    assert semseg is not None
    class_ids = semseg.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert class_ids.shape == (1080, 1920, 1)
    print("CLASS IDS MAX: ", class_ids.max())
    print("CLASS IDS MIN: ", class_ids.min())

    assert np.all(np.logical_and(np.unique(class_ids) <= 256, np.unique(class_ids) >= 0))


def test_decode_file_path(dgp_dataset_train_scene: Scene):
    camera_frame = next(dgp_dataset_train_scene.camera_frames)
    file_path = camera_frame.get_file_path(data_type=AnnotationTypes.SemanticSegmentation2D)
    print("filepath: ", file_path)
    assert file_path is not None
