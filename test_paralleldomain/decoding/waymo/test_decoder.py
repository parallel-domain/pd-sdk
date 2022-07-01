import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.waymo.decoder import WaymoOpenDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.scene import UnorderedScene
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from test_paralleldomain.decoding.constants import WAYMO_OPEN_DATASET_PATH_ENV


@pytest.fixture
def waymo_dataset_path() -> str:
    if WAYMO_OPEN_DATASET_PATH_ENV in os.environ:
        return os.environ[WAYMO_OPEN_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def waymo_train_dataset(waymo_dataset_path: str) -> Dataset:
    decoder = WaymoOpenDatasetDecoder(dataset_path=waymo_dataset_path, split_name="training")
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def waymo_dataset_train_scene(waymo_train_dataset: Dataset) -> UnorderedScene:
    scene_names = waymo_train_dataset.unordered_scene_names
    scene = waymo_train_dataset.get_unordered_scene(scene_name=scene_names[1])
    return scene


def test_can_load_scene(waymo_dataset_train_scene: UnorderedScene):
    assert waymo_dataset_train_scene is not None


def test_knows_all_frames(waymo_dataset_train_scene: UnorderedScene):
    assert (
        len(waymo_dataset_train_scene.frame_ids) == 199
    )  # numer of frames mentioned here https://blog.waymo.com/2019/08/waymo-open-dataset-sharing-our-self.html


def test_knows_all_cameras(waymo_dataset_train_scene: UnorderedScene):
    camera_names = waymo_dataset_train_scene.camera_names
    assert len(camera_names) == 5
    # names: front, front left, front right, side left, side right (https://waymo.com/open/data/perception/)


def test_decode_train_scene_names(waymo_train_dataset: Dataset):
    scene_names = waymo_train_dataset.scene_names
    unordered_scene_names = waymo_train_dataset.unordered_scene_names
    assert len(scene_names) == 0
    assert len(unordered_scene_names) == 798
    # assert len(scene_names) == 1000


def test_decode_frame_camera_names(waymo_dataset_train_scene: UnorderedScene):
    frame_ids = list(waymo_dataset_train_scene.frame_ids)[:5]
    for frame_id in frame_ids:
        frame = waymo_dataset_train_scene.get_frame(frame_id=frame_id)
        names = frame.camera_names
        assert names is not None
        assert len(names) == 5


def test_decode_camera_image(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    for _ in range(5):
        camera_frame = next(camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        image = camera_frame.image
        assert image is not None
        rgb = image.rgb
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (1280, 1920, 3) or rgb.shape == (886, 1920, 3)
        assert rgb.shape[0] == image.height
        assert rgb.shape[1] == image.width
        assert rgb.shape[2] == image.channels


def test_decode_camera_datetime(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    for i in range(5):
        camera_frame = next(camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        date_time = camera_frame.date_time
        assert date_time is not None
        assert date_time > date_time.min


def test_decode_camera_semseg_2d(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    found_labels = 0
    for camera_frame in camera_frames:
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)

        if AnnotationTypes.SemanticSegmentation2D in camera_frame.available_annotation_types:
            found_labels += 1
            semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
            assert semseg is not None
            class_ids = semseg.class_ids
            assert isinstance(class_ids, np.ndarray)
            assert class_ids.shape == (1280, 1920, 1) or class_ids.shape == (886, 1920, 1)
            assert np.all(np.logical_and(np.unique(class_ids) <= 29, np.unique(class_ids) > -1))
    assert found_labels >= 0


def test_decode_camera_isntance_seg_2d(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    found_labels = 0
    for camera_frame in camera_frames:
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)

        if AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types:
            found_labels += 1
            semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
            assert semseg is not None
            instance_ids = semseg.instance_ids
            assert isinstance(instance_ids, np.ndarray)
            assert instance_ids.shape == (1280, 1920, 1) or instance_ids.shape == (886, 1920, 1)
            assert np.all(np.unique(instance_ids) > -1)
    assert found_labels >= 0


def test_decode_class_maps(waymo_dataset_train_scene: UnorderedScene):
    class_maps = waymo_dataset_train_scene.class_maps
    assert len(class_maps) == 1
    assert AnnotationTypes.SemanticSegmentation2D in class_maps
    assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 29


def test_decode_sensor_frame_class_maps(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    found_labels = 0
    for camera_frame in camera_frames:
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)

        if AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types:
            found_labels += 1
            class_maps = camera_frame.class_maps
            assert len(class_maps) == 1
            assert AnnotationTypes.SemanticSegmentation2D in class_maps
            assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 29
    assert found_labels >= 0
