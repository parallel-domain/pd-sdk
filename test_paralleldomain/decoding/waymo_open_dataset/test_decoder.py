import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.waymo_open_dataset.common import WAYMO_USE_ALL_LIDAR_NAME
from paralleldomain.decoding.waymo_open_dataset.decoder import WaymoOpenDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox3D, BoundingBoxes3D
from paralleldomain.model.scene import UnorderedScene
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame, LidarSensor, LidarSensorFrame
from paralleldomain.utilities.transformation import Transformation
from test_paralleldomain.decoding.constants import WAYMO_OPEN_DATASET_PATH_ENV


@pytest.fixture
def waymo_dataset_path() -> str:
    if WAYMO_OPEN_DATASET_PATH_ENV in os.environ:
        return os.environ[WAYMO_OPEN_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def waymo_train_dataset(waymo_dataset_path: str) -> Dataset:
    decoder = WaymoOpenDatasetDecoder(dataset_path=waymo_dataset_path, split_name="training", use_all_lidar=True)
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
    assert len(camera_names) == 6
    # names: front, front left, front right, side left, side right (https://waymo.com/open/data/perception/)


def test_knows_all_lidar_aggregated(waymo_dataset_train_scene: UnorderedScene):
    lidar_names = waymo_dataset_train_scene.lidar_names
    assert len(lidar_names) == 1
    # assert WAYMO_USE_ALL_LIDAR_NAME in lidar_names


def test_decode_train_scene_names(waymo_train_dataset: Dataset):
    scene_names = waymo_train_dataset.scene_names
    unordered_scene_names = waymo_train_dataset.unordered_scene_names
    assert len(scene_names) == 798
    assert len(unordered_scene_names) == 798
    # assert len(scene_names) == 1000


def test_decode_frame_camera_names(waymo_dataset_train_scene: UnorderedScene):
    frame_ids = list(waymo_dataset_train_scene.frame_ids)[:5]
    for frame_id in frame_ids:
        frame = waymo_dataset_train_scene.get_frame(frame_id=frame_id)
        names = frame.camera_names
        assert names is not None
        assert len(names) == 5


def test_decode_frame_lidar_names(waymo_dataset_train_scene: UnorderedScene):
    frame_ids = list(waymo_dataset_train_scene.frame_ids)[:5]
    for frame_id in frame_ids:
        frame = waymo_dataset_train_scene.get_frame(frame_id=frame_id)
        names = frame.lidar_names
        assert names is not None
        assert len(names) == 1


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


def test_decode_lidar_point_cloud(waymo_dataset_train_scene: UnorderedScene):
    lidar_frames = waymo_dataset_train_scene.lidar_frames
    for _ in range(5):
        lidar_frame = next(lidar_frames)
        assert lidar_frame is not None
        assert isinstance(lidar_frame, LidarSensorFrame)
        point_cloud = lidar_frame.point_cloud
        assert point_cloud.xyz.shape[0] > 0
        assert point_cloud.xyz.shape[1] == 3
        assert point_cloud.intensity.shape[1] == 1
        assert point_cloud.elongation.shape[1] == 1


def test_decode_camera_datetime(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    for i in range(5):
        camera_frame = next(camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        date_time = camera_frame.date_time
        assert date_time is not None
        assert date_time > date_time.min


def test_decode_lidar_datetime(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.lidar_frames
    for i in range(5):
        lidar_frame = next(camera_frames)
        assert lidar_frame is not None
        assert isinstance(lidar_frame, LidarSensorFrame)
        date_time = lidar_frame.date_time
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
        if found_labels > 3:
            break
    assert found_labels >= 0


def test_decode_camera_instance_seg_2d(waymo_dataset_train_scene: UnorderedScene):
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
        if found_labels > 3:
            break
    assert found_labels >= 0


def test_decode_lidar_bounding_boxes_3d(waymo_dataset_train_scene: UnorderedScene):
    lidar_frames = waymo_dataset_train_scene.lidar_frames
    found_labels = 0
    for lidar_frame in lidar_frames:
        assert lidar_frame is not None
        assert isinstance(lidar_frame, LidarSensorFrame)

        if AnnotationTypes.BoundingBoxes3D in lidar_frame.available_annotation_types:
            found_labels += 1
            bboxes = lidar_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
            assert bboxes is not None
            assert isinstance(bboxes, BoundingBoxes3D)
            assert isinstance(bboxes.boxes[0], BoundingBox3D)
            assert isinstance(bboxes.boxes[0].pose, Transformation)
            class_ids = [box.class_id > -1 for box in bboxes.boxes]
            assert all(class_ids)
        if found_labels > 3:
            break
    assert found_labels >= 0


def test_decode_class_maps(waymo_dataset_train_scene: UnorderedScene):
    class_maps = waymo_dataset_train_scene.class_maps
    assert len(class_maps) == 3
    assert AnnotationTypes.BoundingBoxes2D in class_maps
    assert AnnotationTypes.SemanticSegmentation2D in class_maps
    assert AnnotationTypes.BoundingBoxes3D in class_maps
    assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 29
    assert len(class_maps[AnnotationTypes.BoundingBoxes2D].class_names) == 3
    assert len(class_maps[AnnotationTypes.BoundingBoxes3D].class_names) == 5


def test_decode_sensor_frame_class_maps(waymo_dataset_train_scene: UnorderedScene):
    camera_frames = waymo_dataset_train_scene.camera_frames
    found_labels = 0
    for camera_frame in camera_frames:
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)

        if AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types:
            found_labels += 1
            class_maps = camera_frame.class_maps
            assert len(class_maps) == 3
            assert AnnotationTypes.SemanticSegmentation2D in class_maps
            assert AnnotationTypes.BoundingBoxes2D in class_maps
            assert AnnotationTypes.BoundingBoxes3D in class_maps
            assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 29
            assert len(class_maps[AnnotationTypes.BoundingBoxes2D].class_names) == 3
            assert len(class_maps[AnnotationTypes.BoundingBoxes3D].class_names) == 5
        if found_labels > 3:
            break
    assert found_labels >= 0
