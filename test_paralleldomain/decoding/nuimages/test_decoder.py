import os

import numpy as np

from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensor

NUIMAGES_PATH_ENV = "NUIMAGES_PATH_ENV"


def test_decode_test_scene_names():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-test")
        dataset = decoder.get_dataset()
        assert len(dataset.scene_names) == 60


def test_decode_train_scene_names():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-train")
        dataset = decoder.get_dataset()
        assert len(dataset.scene_names) == 350


def test_decode_sensor_names():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.scene_names
        scene = dataset.get_scene(scene_name=scene_names[0])
        sensor_names = scene.sensor_names
        assert len(sensor_names) == 6
        camera_names = scene.camera_names
        assert len(camera_names) == 6
        lidar_names = scene.lidar_names
        assert len(lidar_names) == 0


def test_decode_frame_ids():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        assert len(scene.frame_ids) > 0


def test_decode_camera_frame_ids():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
        assert len(camera.frame_ids) > 0


def test_decode_frame_ego_pose():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.scene_names
        scene = dataset.get_scene(scene_name=scene_names[0])
        frame = scene.get_frame(frame_id=scene.frame_ids[-1])
        ego_frame = frame.ego_frame
        assert ego_frame is not None
        assert isinstance(ego_frame.pose, np.ndarray)
        assert np.any(ego_frame.pose.as_euler_angles(order="XYZ") > 0.0)


def test_decode_available_annotation_types():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        assert len(scene.available_annotation_types) == 2
        assert AnnotationTypes.SemanticSegmentation2D in scene.available_annotation_types
        assert AnnotationTypes.InstanceSegmentation2D in scene.available_annotation_types


def test_decode_camera():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
        assert isinstance(camera, CameraSensor)


def test_decode_camera_intrinsic():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.scene_names
        scene = dataset.get_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
        assert isinstance(camera, CameraSensor)
        cam_frame = camera.get_frame(frame_id=list(camera.frame_ids)[-1])
        intrinsic = cam_frame.intrinsic
        assert intrinsic.fx != 0.0


def test_decode_camera_image():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
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


def test_decode_camera_semseg_2d():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
        assert camera is not None
        assert isinstance(camera, CameraSensor)
        camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
        semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        assert semseg is not None
        class_ids = semseg.class_ids
        assert isinstance(class_ids, np.ndarray)
        assert class_ids.shape == (1024, 2048, 1)
        assert np.all(np.logical_and(np.unique(class_ids) <= 33, np.unique(class_ids) > -1))


def test_decode_camera_instance_seg_2d():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_unordered_scene(scene_name=scene_names[0])
        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])
        assert camera is not None
        assert isinstance(camera, CameraSensor)
        camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[0])
        instanceseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
        assert instanceseg is not None
        instance_ids = instanceseg.instance_ids
        assert isinstance(instance_ids, np.ndarray)
        assert instance_ids.shape == (1024, 2048, 1)
        assert len(np.unique(instance_ids) < 19) > 0


def test_decode_class_maps():
    if NUIMAGES_PATH_ENV in os.environ:
        nuimages_path = os.environ[NUIMAGES_PATH_ENV]
        decoder = NuImagesDatasetDecoder(dataset_path=nuimages_path, split_name="v1.0-mini")
        dataset = decoder.get_dataset()
        scene_names = dataset.unordered_scene_names
        scene = dataset.get_scene(scene_name=scene_names[0])
        class_maps = scene.class_maps
        assert len(class_maps) == 1
        assert AnnotationTypes.SemanticSegmentation2D in class_maps
        assert len(class_maps.keys()) == 35
