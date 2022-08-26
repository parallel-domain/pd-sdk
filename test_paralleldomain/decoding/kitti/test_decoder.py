import os
from typing import List

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.kitti_flow.decoder import KITTIFlowDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from test_paralleldomain.decoding.constants import KITTI_FLOW_DATASET_PATH_ENV


@pytest.fixture
def kitti_dataset_path() -> str:
    if KITTI_FLOW_DATASET_PATH_ENV in os.environ:
        return os.environ[KITTI_FLOW_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def kitti_train_dataset(kitti_dataset_path: str) -> Dataset:
    decoder = KITTIFlowDatasetDecoder(dataset_path=kitti_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def kitti_dataset_train_scenes(kitti_train_dataset: Dataset) -> List:
    scene_names = kitti_train_dataset.scene_names
    return scene_names


@pytest.fixture()
def kitti_dataset_train_scene(kitti_train_dataset: Dataset, kitti_dataset_train_scenes: List) -> Scene:
    scene = kitti_train_dataset.get_scene(scene_name=kitti_dataset_train_scenes[0])
    return scene


def test_can_load_scene(kitti_dataset_train_scenes: List, kitti_train_dataset: Dataset):
    for scene_name in kitti_dataset_train_scenes:
        scene = kitti_train_dataset.get_scene(scene_name=scene_name)
        assert scene is not None


def test_knows_all_frames(kitti_dataset_train_scenes: List, kitti_train_dataset: Dataset):
    for scene_name in kitti_dataset_train_scenes:
        scene = kitti_train_dataset.get_scene(scene_name=scene_name)
        assert len(scene.frame_ids) == 2  # Each scene is a pair of first/second frames, should always be 2.


def test_decode_train_scene_names(kitti_train_dataset: Dataset):
    assert len(kitti_train_dataset.scene_names) == 200
    assert len(kitti_train_dataset.unordered_scene_names) == 200


def test_decode_camera_image(kitti_dataset_train_scene: Scene):
    camera_frame = next(kitti_dataset_train_scene.camera_frames)
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    image = camera_frame.image
    assert image is not None
    rgb = image.rgb
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[0] > 350
    assert rgb.shape[1] > 1200
    assert rgb.shape[2] == 3
    assert rgb.shape[0] == image.height
    assert rgb.shape[1] == image.width
    assert rgb.shape[2] == image.channels


def test_decode_optical_flow(kitti_dataset_train_scene: Scene):
    camera_frame = kitti_dataset_train_scene.get_frame(frame_id=kitti_dataset_train_scene.frame_ids[0]).get_sensor(
        sensor_name="default"
    )
    assert camera_frame is not None
    assert isinstance(camera_frame, CameraSensorFrame)
    if AnnotationTypes.OpticalFlow in camera_frame.available_annotation_types:
        flow = camera_frame.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)
        assert flow is not None
        vectors = flow.vectors
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape[0] > 350
        assert vectors.shape[1] > 1200
        assert vectors.shape[2] == 2
        assert np.all(np.logical_and(flow.vectors.max() < 512, flow.vectors.min() >= -512))


def test_decode_camera_image_next(kitti_dataset_train_scene: Scene):
    dataset_path = kitti_dataset_train_scene.metadata["dataset_path"]
    image_folder = kitti_dataset_train_scene.metadata["image_folder"]
    camera_frame = next(kitti_dataset_train_scene.camera_frames)
    image = camera_frame.image
    fid = camera_frame.frame_id
    fid_next = fid[:-5] + "1.png"
    next_image_path = AnyPath(dataset_path) / image_folder / fid_next
    next_image = read_image(next_image_path)
    assert next_image is not None
    assert isinstance(next_image, np.ndarray)
    assert next_image.shape[0] > 350
    assert next_image.shape[1] > 1200
    assert next_image.shape[2] == 3
    assert next_image.shape[0] == image.height
    assert next_image.shape[1] == image.width
    assert next_image.shape[2] == image.channels
