import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.cityscapes.decoder import CityscapesDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensor
from paralleldomain.model.unordered_scene import UnorderedScene
from test_paralleldomain.decoding.constants import CITYSCAPES_DATASET_PATH_ENV


@pytest.fixture
def cityscapes_dataset_path() -> str:
    if CITYSCAPES_DATASET_PATH_ENV in os.environ:
        return os.environ[CITYSCAPES_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def cityscapes_test_dataset(cityscapes_dataset_path: str) -> Dataset:
    decoder = CityscapesDatasetDecoder(dataset_path=cityscapes_dataset_path, splits=["test"])
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture
def cityscapes_train_dataset(cityscapes_dataset_path: str) -> Dataset:
    decoder = CityscapesDatasetDecoder(dataset_path=cityscapes_dataset_path, splits=["train"])
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def cityscapes_first_train_scene(cityscapes_train_dataset: Dataset) -> UnorderedScene:
    scene_names = cityscapes_train_dataset.unordered_scene_names
    scene = cityscapes_train_dataset.get_unordered_scene(scene_name=scene_names[0])
    return scene


def test_decode_test_scene_names(cityscapes_test_dataset: Dataset):
    assert len(cityscapes_test_dataset.scene_names) == 0
    scene_names = cityscapes_test_dataset.unordered_scene_names
    test_scenes = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
    assert len(test_scenes) == len(scene_names)
    for scene_name in test_scenes:
        assert f"test-{scene_name}" in scene_names


def test_decode_train_scene_names(cityscapes_train_dataset: Dataset):
    assert len(cityscapes_train_dataset.scene_names) == 0
    scene_names = cityscapes_train_dataset.unordered_scene_names
    train_scenes = [
        "aachen",
        "bochum",
        "bremen",
        "cologne",
        "darmstadt",
        "dusseldorf",
        "erfurt",
        "hamburg",
        "hanover",
        "jena",
        "krefeld",
        "monchengladbach",
        "strasbourg",
        "stuttgart",
        "tubingen",
        "ulm",
        "weimar",
        "zurich",
    ]
    assert len(train_scenes) == len(scene_names)
    for scene_name in train_scenes:
        assert f"train-{scene_name}" in scene_names


def test_decode_sensor_names(cityscapes_first_train_scene: UnorderedScene):
    sensor_names = cityscapes_first_train_scene.sensor_names
    assert len(sensor_names) == 1
    camera_names = cityscapes_first_train_scene.camera_names
    assert len(camera_names) == 1
    lidar_names = cityscapes_first_train_scene.lidar_names
    assert len(lidar_names) == 0


def test_decode_class_maps(cityscapes_first_train_scene: UnorderedScene):
    class_maps = cityscapes_first_train_scene.class_maps
    assert len(class_maps) == 1
    assert AnnotationTypes.SemanticSegmentation2D in class_maps
    assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 35


def test_decode_frame_ids(cityscapes_first_train_scene: UnorderedScene):
    assert len(cityscapes_first_train_scene.frame_ids) > 0


def test_decode_camera_frame_ids(cityscapes_first_train_scene: UnorderedScene):
    camera = cityscapes_first_train_scene.get_camera_sensor(camera_name=cityscapes_first_train_scene.camera_names[0])
    assert len(camera.frame_ids) > 0


def test_decode_available_annotation_types(cityscapes_first_train_scene: UnorderedScene):
    assert len(cityscapes_first_train_scene.available_annotation_types) == 2
    assert AnnotationTypes.SemanticSegmentation2D in cityscapes_first_train_scene.available_annotation_types
    assert AnnotationTypes.InstanceSegmentation2D in cityscapes_first_train_scene.available_annotation_types


def test_decode_camera(cityscapes_first_train_scene: UnorderedScene):
    camera = cityscapes_first_train_scene.get_camera_sensor(camera_name=cityscapes_first_train_scene.camera_names[0])
    assert isinstance(camera, CameraSensor)


def test_decode_camera_image(cityscapes_first_train_scene: UnorderedScene):
    camera = cityscapes_first_train_scene.get_camera_sensor(camera_name=cityscapes_first_train_scene.camera_names[0])
    assert camera is not None
    assert isinstance(camera, CameraSensor)
    camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
    image = camera_frame.image
    assert image is not None
    rgb = image.rgb
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (1024, 2048, 3)
    assert rgb.shape[0] == image.height
    assert rgb.shape[1] == image.width
    assert rgb.shape[2] == image.channels


def test_decode_camera_semseg_2d(cityscapes_first_train_scene: UnorderedScene):
    camera = cityscapes_first_train_scene.get_camera_sensor(camera_name=cityscapes_first_train_scene.camera_names[0])
    assert camera is not None
    assert isinstance(camera, CameraSensor)
    camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
    semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    assert semseg is not None
    class_ids = semseg.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert class_ids.shape == (1024, 2048, 1)
    assert np.all(np.logical_and(np.unique(class_ids) <= 33, np.unique(class_ids) > -1))


def test_decode_camera_instance_seg_2d(cityscapes_first_train_scene: UnorderedScene):
    camera = cityscapes_first_train_scene.get_camera_sensor(camera_name=cityscapes_first_train_scene.camera_names[0])
    assert camera is not None
    assert isinstance(camera, CameraSensor)
    camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
    instanceseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
    assert instanceseg is not None
    instance_ids = instanceseg.instance_ids
    assert isinstance(instance_ids, np.ndarray)
    assert instance_ids.shape == (1024, 2048, 1)
    assert len(np.unique(instance_ids) < 19) > 0
