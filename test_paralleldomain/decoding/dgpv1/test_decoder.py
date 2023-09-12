import os
from typing import List

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from test_paralleldomain.decoding.constants import DGP_V1_DATASET_PATH_ENV


@pytest.fixture
def dgpv1_dataset_path() -> str:
    if DGP_V1_DATASET_PATH_ENV in os.environ:
        return os.environ[DGP_V1_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def dgpv1_train_dataset(dgpv1_dataset_path: str) -> Dataset:
    decoder = DGPDatasetDecoder(dataset_path=dgpv1_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def dgpv1_dataset_train_scenes(dgpv1_train_dataset: Dataset) -> List:
    scene_names = dgpv1_train_dataset.scene_names
    return scene_names


@pytest.fixture()
def dgpv1_dataset_train_scene(dgpv1_train_dataset: Dataset, dgpv1_dataset_train_scenes: List) -> Scene:
    scene = dgpv1_train_dataset.get_scene(scene_name=dgpv1_dataset_train_scenes[0])
    return scene


def test_can_load_scene(dgpv1_dataset_train_scenes: List, dgpv1_train_dataset: Dataset):
    for scene_name in dgpv1_dataset_train_scenes:
        scene = dgpv1_train_dataset.get_scene(scene_name=scene_name)
        assert scene is not None


def test_knows_all_frames(dgpv1_dataset_train_scenes: List, dgpv1_train_dataset: Dataset):
    for scene_name in dgpv1_dataset_train_scenes:
        scene = dgpv1_train_dataset.get_scene(scene_name=scene_name)
        assert len(scene.frame_ids) > 0


def test_decode_train_scene_names(dgpv1_train_dataset: Dataset):
    assert len(dgpv1_train_dataset.scene_names) == 46
    assert len(dgpv1_train_dataset.unordered_scene_names) == 46


def test_decode_camera_image(dgpv1_dataset_train_scene: Scene):
    camera_frame = next(dgpv1_dataset_train_scene.camera_frames)
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    image = camera_frame.image
    assert image is not None
    rgb = image.rgb
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[0] == 900
    assert rgb.shape[1] == 1600
    assert rgb.shape[2] in [3, 4]
    assert rgb.shape[0] == image.height
    assert rgb.shape[1] == image.width


def test_decode_camera_semseg_2d(dgpv1_dataset_train_scene: Scene):
    camera_frame = next(dgpv1_dataset_train_scene.camera_frames)
    semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    assert semseg is not None
    class_ids = semseg.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert class_ids.shape == (900, 1600, 1)
    assert np.all(np.logical_and(np.unique(class_ids) <= 31, np.unique(class_ids) >= 0))


def test_decode_file_path(dgpv1_dataset_train_scene: Scene):
    camera_frame = next(dgpv1_dataset_train_scene.camera_frames)
    file_path = camera_frame.get_file_path(data_type=AnnotationTypes.SemanticSegmentation2D)
    print("filepath: ", file_path)
    assert file_path is not None
