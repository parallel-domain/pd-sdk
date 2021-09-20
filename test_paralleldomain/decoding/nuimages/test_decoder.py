import os

import numpy as np
import pytest

from paralleldomain import Scene
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame

NUIMAGES_PATH_ENV = "NUIMAGES_PATH_ENV"


class TestDataset:
    def test_decode_test_scene_names(self):
        if NUIMAGES_PATH_ENV in os.environ:
            nuimages_path = os.environ[NUIMAGES_PATH_ENV]
            decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-val")
            dataset = decoder.get_dataset()
            assert len(dataset.scene_names) == 82

    def test_decode_train_scene_names(self):
        if NUIMAGES_PATH_ENV in os.environ:
            nuimages_path = os.environ[NUIMAGES_PATH_ENV]
            decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-train")
            dataset = decoder.get_dataset()
            assert len(dataset.scene_names) == 350


@pytest.fixture
def two_cam_scene() -> Scene:
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-val")
        dataset = decoder.get_dataset()
        return dataset.get_scene(scene_name="c6b4b836e4c543378e340a8a28760ebd")


class TestScene:
    def test_decode_sensor_names(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            sensor_names = two_cam_scene.sensor_names
            assert len(sensor_names) == 6
            camera_names = two_cam_scene.camera_names
            for cam_name in [
                "CAM_FRONT_LEFT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT",
                "CAM_BACK_LEFT",
            ]:
                assert cam_name in camera_names
                assert cam_name in sensor_names
            assert len(camera_names) == 6
            lidar_names = two_cam_scene.lidar_names
            assert len(lidar_names) == 0

    def test_decode_frame_ids(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            assert len(two_cam_scene.frame_ids) == 126

    def test_decode_available_annotation_types(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            assert len(two_cam_scene.available_annotation_types) == 3
            assert AnnotationTypes.SemanticSegmentation2D in two_cam_scene.available_annotation_types
            assert AnnotationTypes.InstanceSegmentation2D in two_cam_scene.available_annotation_types
            assert AnnotationTypes.BoundingBoxes2D in two_cam_scene.available_annotation_types

    def test_decode_camera(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            for cam_name in two_cam_scene.camera_names:
                camera = two_cam_scene.get_camera_sensor(camera_name=cam_name)
                assert isinstance(camera, CameraSensor)

    def test_decode_class_maps(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            class_maps = two_cam_scene.class_maps
            assert len(class_maps) == 2
            assert AnnotationTypes.SemanticSegmentation2D in class_maps
            assert AnnotationTypes.BoundingBoxes2D in class_maps
            assert len(class_maps[AnnotationTypes.SemanticSegmentation2D].class_names) == 25
            assert len(class_maps[AnnotationTypes.BoundingBoxes2D].class_names) == 25


class TestCamera:
    def test_decode_camera_frame_ids(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
            assert len(camera.frame_ids) == 22

    def test_decode_camera_intrinsic(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
            assert isinstance(camera, CameraSensor)
            cam_frame = camera.get_frame(frame_id=list(camera.frame_ids)[-1])
            intrinsic = cam_frame.intrinsic
            assert intrinsic.fx != 0.0


class TestFrame:
    def test_decode_frame_ego_pose(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            nuimages_path = os.environ[NUIMAGES_PATH_ENV]
            decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
            dataset = decoder.get_dataset()
            scene_names = dataset.scene_names
            scene = dataset.get_scene(scene_name=scene_names[0])
            frame = scene.get_frame(frame_id=scene.frame_ids[-1])
            ego_frame = frame.ego_frame
            assert ego_frame is not None
            assert isinstance(ego_frame.pose.transformation_matrix, np.ndarray)
            assert np.any(ego_frame.pose.as_euler_angles(order="XYZ") > 0.0)

    def test_decode_frame_sensor_names(self, two_cam_scene: Scene):
        if NUIMAGES_PATH_ENV in os.environ:
            frame = two_cam_scene.get_frame(frame_id=two_cam_scene.frame_ids[0])
            camera_names = frame.camera_names
            assert camera_names is not None
            assert len(camera_names) == 1


@pytest.fixture()
def some_camera_frame(two_cam_scene: Scene) -> CameraSensorFrame:
    camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
    assert camera is not None
    assert isinstance(camera, CameraSensor)
    camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
    return camera_frame


class TestCameraSensorFrame:
    def test_decode_camera_image(self, some_camera_frame: CameraSensorFrame):
        if NUIMAGES_PATH_ENV in os.environ:
            image = some_camera_frame.image
            assert image is not None
            rgb = image.rgb
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (900, 1600, 3)
            assert rgb.shape[0] == image.height
            assert rgb.shape[1] == image.width
            assert rgb.shape[2] == image.channels

    def test_decode_camera_semseg_2d(self, some_camera_frame: CameraSensorFrame):
        if NUIMAGES_PATH_ENV in os.environ:
            semseg = some_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
            assert semseg is not None
            class_ids = semseg.class_ids
            assert isinstance(class_ids, np.ndarray)
            assert class_ids.shape == (900, 1600, 1)
            assert np.all(np.logical_and(np.unique(class_ids) <= 31, np.unique(class_ids) >= 0))

    def test_decode_camera_instance_seg_2d(self, some_camera_frame: CameraSensorFrame):
        if NUIMAGES_PATH_ENV in os.environ:
            instanceseg = some_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
            assert instanceseg is not None
            instance_ids = instanceseg.instance_ids
            assert isinstance(instance_ids, np.ndarray)
            assert instance_ids.shape == (900, 1600, 1)
            assert len(np.unique(instance_ids) > 0) > 0

    def test_decode_camera_bbox_2d(self, some_camera_frame: CameraSensorFrame):
        if NUIMAGES_PATH_ENV in os.environ:
            boxes = some_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
            assert boxes is not None
            boxes = boxes.boxes
            assert len(boxes) > 0
            for box in boxes:
                assert isinstance(box, BoundingBox2D)
                assert box.area > 0
