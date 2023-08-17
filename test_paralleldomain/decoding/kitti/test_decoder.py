import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.kitti.decoder import KittiDatasetDecoder, KITTI_CLASS_MAP
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox3D
from paralleldomain.model.sensor import LidarSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from test_paralleldomain.decoding.constants import KITTI_DATASET_PATH_ENV


@pytest.fixture
def kitti_dataset_path() -> str:
    if KITTI_DATASET_PATH_ENV in os.environ:
        return os.environ[KITTI_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def kitti_train_dataset(kitti_dataset_path: str) -> Dataset:
    decoder = KittiDatasetDecoder(dataset_path=kitti_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def kitti_dataset_train_scene(kitti_train_dataset: Dataset) -> UnorderedScene:
    scene_names = kitti_train_dataset.unordered_scene_names
    scene = kitti_train_dataset.get_unordered_scene(scene_name=scene_names[0])
    return scene


def test_can_load_scene(kitti_dataset_train_scene: UnorderedScene):
    assert kitti_dataset_train_scene is not None


def test_knows_all_frames(kitti_dataset_train_scene: UnorderedScene):
    assert (
        # kitti train is 7481
        len(kitti_dataset_train_scene.frame_ids)
        == 7481
    )


def test_decode_train_scene_names(kitti_train_dataset: Dataset):
    assert len(kitti_train_dataset.scene_names) == 0
    scene_names = kitti_train_dataset.unordered_scene_names
    assert 1 == len(scene_names)


def test_decode_lidar_point_cloud(kitti_dataset_train_scene: UnorderedScene):
    lidar_frames = kitti_dataset_train_scene.lidar_frames
    for i in range(5):
        lidar_frame = next(lidar_frames)
        assert lidar_frame is not None
        assert isinstance(lidar_frame, LidarSensorFrame)
        point_cloud = lidar_frame.point_cloud
        assert point_cloud is not None
        xyz = point_cloud.xyz
        intensity = point_cloud.intensity
        assert isinstance(xyz, np.ndarray)
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        assert xyz.shape[0] > 1e5
        assert isinstance(intensity, np.ndarray)
        assert len(intensity.shape) == 1 or intensity.shape[1] == 1


def test_decode_lidar_object_detection_3d(kitti_dataset_train_scene: UnorderedScene):
    lidar_frames = kitti_dataset_train_scene.lidar_frames
    for i in range(5):
        lidar_frame = next(lidar_frames)
        assert lidar_frame is not None
        assert isinstance(lidar_frame, LidarSensorFrame)
        bboxes = lidar_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert bboxes is not None
        boxes = bboxes.boxes
        assert isinstance(boxes, list)
        assert isinstance(boxes[0], BoundingBox3D)
        class_ids = [box.class_id for box in boxes]
        assert all([cid < 15 for cid in class_ids])


def test_decode_class_maps(kitti_dataset_train_scene: UnorderedScene):
    class_maps = kitti_dataset_train_scene.class_maps
    assert len(class_maps) == 1
    assert AnnotationTypes.BoundingBoxes3D in class_maps
    assert len(class_maps[AnnotationTypes.BoundingBoxes3D].class_names) == len(KITTI_CLASS_MAP)
