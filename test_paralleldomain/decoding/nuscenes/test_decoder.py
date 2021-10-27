import base64
import os

import numpy as np
import pytest

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.nuscenes.decoder import NuScenesDatasetDecoder
from paralleldomain.decoding.nuscenes.sensor_frame_decoder import mask_decode
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox3D
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame, LidarSensor, LidarSensorFrame
from paralleldomain.utilities.transformation import Transformation
from test_paralleldomain.decoding.constants import NUSCENES_MINI_DATASET_PATH_ENV, NUSCENES_TRAINVAL_DATASET_PATH_ENV


@pytest.fixture
def nuscenes_trainval_dataset_path() -> str:
    if NUSCENES_TRAINVAL_DATASET_PATH_ENV in os.environ:
        return os.environ[NUSCENES_TRAINVAL_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def nuscenes_mini_dataset_path() -> str:
    if NUSCENES_MINI_DATASET_PATH_ENV in os.environ:
        return os.environ[NUSCENES_MINI_DATASET_PATH_ENV]
    else:
        pytest.skip()


@pytest.fixture
def nuscenes_trainval_dataset(nuscenes_trainval_dataset_path: str) -> Dataset:
    decoder = NuScenesDatasetDecoder(dataset_path=nuscenes_trainval_dataset_path, split_name="v1.0-trainval")
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture
def nuscenes_mini_dataset(nuscenes_mini_dataset_path: str) -> Dataset:
    decoder = NuScenesDatasetDecoder(dataset_path=nuscenes_mini_dataset_path, split_name="v1.0-mini")
    dataset = decoder.get_dataset()
    return dataset


class TestDataset:
    def test_decode_trainval_scene_names(self, nuscenes_trainval_dataset: Dataset):
        assert len(nuscenes_trainval_dataset.scene_names) == 850

    def test_decode_mini_scene_names(self, nuscenes_mini_dataset: Dataset):
        assert len(nuscenes_mini_dataset.scene_names) == 10


@pytest.fixture
def two_cam_scene(nuscenes_mini_dataset: Dataset) -> Scene:
    return nuscenes_mini_dataset.get_scene(scene_name="fcbccedd61424f1b85dcbf8f897f9754")


class TestScene:
    def test_decode_sensor_names(self, two_cam_scene: Scene):
        sensor_names = two_cam_scene.sensor_names
        assert len(sensor_names) == 7
        camera_names = two_cam_scene.camera_names
        lidar_names = two_cam_scene.lidar_names
        for cam_name in [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]:
            assert cam_name in camera_names
            assert cam_name in sensor_names
        for lidar_name in ["LIDAR_TOP"]:
            assert lidar_name in lidar_names
            assert lidar_name in sensor_names
        assert len(camera_names) == 6
        assert len(lidar_names) == 1

    def test_decode_frame_ids(self, two_cam_scene: Scene):
        assert len(two_cam_scene.frame_ids) == 40

    def test_decode_available_annotation_types(self, two_cam_scene: Scene):
        assert len(two_cam_scene.available_annotation_types) == 1
        assert AnnotationTypes.BoundingBoxes3D in two_cam_scene.available_annotation_types

    def test_decode_camera(self, two_cam_scene: Scene):
        for cam_name in two_cam_scene.camera_names:
            camera = two_cam_scene.get_camera_sensor(camera_name=cam_name)
            assert isinstance(camera, CameraSensor)

    def test_decode_lidar(self, two_cam_scene: Scene):
        for lidar_name in two_cam_scene.lidar_names:
            lidar = two_cam_scene.get_lidar_sensor(lidar_name=lidar_name)
            assert isinstance(lidar, LidarSensor)

    def test_decode_class_maps(self, two_cam_scene: Scene):
        class_maps = two_cam_scene.class_maps
        assert len(class_maps) == 1
        assert AnnotationTypes.BoundingBoxes3D in class_maps
        assert len(class_maps[AnnotationTypes.BoundingBoxes3D].class_names) == 24


class TestCamera:
    def test_decode_camera_frame_ids(self, two_cam_scene: Scene):
        camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
        assert camera.name == "CAM_BACK"
        assert len(camera.frame_ids) == 40

    def test_decode_camera_intrinsic(self, two_cam_scene: Scene):
        camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
        assert isinstance(camera, CameraSensor)
        cam_frame = camera.get_frame(frame_id=list(camera.frame_ids)[-1])
        intrinsic = cam_frame.intrinsic
        assert intrinsic.fx != 0.0


class TestLidar:
    def test_decode_lidar_frame_ids(self, two_cam_scene: Scene):
        lidar = two_cam_scene.get_lidar_sensor(lidar_name=two_cam_scene.lidar_names[0])
        assert lidar.name == "LIDAR_TOP"
        assert len(lidar.frame_ids) == 40

    def test_decode_lidar_transformations(self, two_cam_scene: Scene):
        lidar = two_cam_scene.get_lidar_sensor(lidar_name=two_cam_scene.lidar_names[0])
        assert isinstance(lidar, LidarSensor)
        lidar_frame = lidar.get_frame(frame_id=list(lidar.frame_ids)[-1])
        assert isinstance(lidar_frame.pose, Transformation)
        assert isinstance(lidar_frame.extrinsic, Transformation)


class TestFrame:
    def test_decode_frame_ego_pose(self, nuscenes_mini_dataset: Dataset):
        scene_names = nuscenes_mini_dataset.scene_names
        scene = nuscenes_mini_dataset.get_scene(scene_name=scene_names[0])
        frame = scene.get_frame(frame_id=scene.frame_ids[-1])
        ego_frame = frame.ego_frame
        assert ego_frame is not None
        assert isinstance(ego_frame.pose.transformation_matrix, np.ndarray)
        assert np.any(ego_frame.pose.as_euler_angles(order="XYZ") > 0.0)

    def test_decode_frame_sensor_names(self, two_cam_scene: Scene):
        frame = two_cam_scene.get_frame(frame_id=two_cam_scene.frame_ids[0])
        camera_names = frame.camera_names
        assert camera_names is not None
        assert len(camera_names) == 6


@pytest.fixture()
def some_camera_frame(two_cam_scene: Scene) -> CameraSensorFrame:
    camera = two_cam_scene.get_camera_sensor(camera_name=two_cam_scene.camera_names[0])
    assert camera is not None
    assert isinstance(camera, CameraSensor)
    camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
    return camera_frame


class TestCameraSensorFrame:
    def test_decode_camera_image(self, some_camera_frame: CameraSensorFrame):
        image = some_camera_frame.image
        assert image is not None
        rgb = image.rgb
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (900, 1600, 3)
        assert rgb.shape[0] == image.height
        assert rgb.shape[1] == image.width
        assert rgb.shape[2] == image.channels

    def test_decode_camera_bbox_3d(self, some_camera_frame: CameraSensorFrame):
        boxes = some_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert boxes is not None
        boxes = boxes.boxes
        assert len(boxes) > 0
        for box in boxes:
            assert isinstance(box, BoundingBox3D)
            assert box.volume > 0

    @pytest.mark.skip("For debugging")
    def test_data_loader(self, nuscenes_trainval_dataset: Dataset):
        for scene_name in nuscenes_trainval_dataset.scene_names[:5]:
            scene = nuscenes_trainval_dataset.get_scene(scene_name=scene_name)
            for frame in scene.frames:
                for camera in frame.camera_frames:
                    if AnnotationTypes.BoundingBoxes3D in camera.available_annotation_types:
                        pass

    @pytest.mark.skip("For debugging")
    def test_data_loader_with_data_loading(self, nuscenes_trainval_dataset: Dataset):
        for scene_name in nuscenes_trainval_dataset.scene_names[:2]:
            scene = nuscenes_trainval_dataset.get_scene(scene_name=scene_name)
            for frame in scene.frames[:5]:
                for camera in frame.camera_frames:
                    if AnnotationTypes.BoundingBoxes3D in camera.available_annotation_types:
                        assert camera.image.rgb is not None
                        assert (
                            camera.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).class_ids
                            is not None
                        )


@pytest.fixture()
def some_lidar_frame(two_cam_scene: Scene) -> LidarSensorFrame:
    lidar = two_cam_scene.get_lidar_sensor(lidar_name=two_cam_scene.lidar_names[0])
    assert lidar is not None
    assert isinstance(lidar, LidarSensor)
    lidar_frame = lidar.get_frame(frame_id=list(lidar.frame_ids)[0])
    return lidar_frame


class TestLidarSensorFrame:
    def test_decode_lidar_point_cloud(self, some_lidar_frame: LidarSensorFrame):
        pt_cloud = some_lidar_frame.point_cloud
        assert pt_cloud is not None
        xyz = pt_cloud.xyz
        xyz_i = pt_cloud.xyz_i
        ring = pt_cloud.ring
        intensity = pt_cloud.intensity
        assert isinstance(xyz, np.ndarray)
        assert xyz.sum() - xyz_i[:, :3].sum() == 0.0
        assert ring.max() < 32.0
        assert ring.min() > -1.0
        assert xyz.shape[1] == 3
        assert intensity.max() < 256.0
        assert intensity.min() > -1.0

    def test_decode_lidar_bbox_3d(self, some_lidar_frame: LidarSensorFrame):
        boxes = some_lidar_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert boxes is not None
        boxes = boxes.boxes
        assert len(boxes) > 0
        for box in boxes:
            assert isinstance(box, BoundingBox3D)
            assert box.volume > 0
