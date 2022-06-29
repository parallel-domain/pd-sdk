import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.gta5.decoder import GTADatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from test_paralleldomain.decoding.constants import GTA5_DATASET_PATH_ENV


@pytest.fixture
def gta_dataset_path() -> str:
    if GTA5_DATASET_PATH_ENV in os.environ:
        return os.environ[GTA5_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def gta_train_dataset(gta_dataset_path: str) -> Dataset:
    decoder = GTADatasetDecoder(dataset_path=gta_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def gta_dataset_train_scene(gta_train_dataset: Dataset) -> UnorderedScene:
    scene_names = gta_train_dataset.unordered_scene_names
    scene = gta_train_dataset.get_unordered_scene(scene_name=scene_names[0])
    return scene


def test_can_load_scene(gta_dataset_train_scene: UnorderedScene):
    assert gta_dataset_train_scene is not None


def test_knows_all_frames(gta_dataset_train_scene: UnorderedScene):
    assert (
        len(gta_dataset_train_scene.frame_ids) == 24966
    )  # numer o frames mentioned here https://download.visinf.tu-darmstadt.de/data/from_games/


def test_decode_train_scene_names(gta_train_dataset: Dataset):
    assert len(gta_train_dataset.scene_names) == 0
    scene_names = gta_train_dataset.unordered_scene_names
    assert 1 == len(scene_names)


def test_decode_camera_image(gta_dataset_train_scene: UnorderedScene):
    for i in range(5):
        camera_frame = next(gta_dataset_train_scene.camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        image = camera_frame.image
        assert image is not None
        rgb = image.rgb
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (1024, 2048, 3) or rgb.shape == (1052, 1914, 3)
        assert rgb.shape[0] == image.height
        assert rgb.shape[1] == image.width
        assert rgb.shape[2] == image.channels


def test_decode_camera_semseg_2d(gta_dataset_train_scene: UnorderedScene):
    for i in range(5):
        camera_frame = next(gta_dataset_train_scene.camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        assert semseg is not None
        class_ids = semseg.class_ids
        assert isinstance(class_ids, np.ndarray)
        assert class_ids.shape == (1024, 2048, 1) or class_ids.shape == (1052, 1914, 1)
        assert np.all(np.logical_and(np.unique(class_ids) <= 33, np.unique(class_ids) > -1))


def test_decode_class_maps(gta_dataset_train_scene: UnorderedScene):
    class_maps = gta_dataset_train_scene.class_maps
    assert len(class_maps) == 1
    assert AnnotationTypes.SemanticSegmentation2D in class_maps
    assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 35
