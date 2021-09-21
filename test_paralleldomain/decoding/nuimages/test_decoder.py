import base64
import os

import numpy as np
import pytest

from paralleldomain import Scene
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.decoding.nuimages.sensor_frame_decoder import mask_decode
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


# keep this for debugging
# def test_decode_visual():
#     s = "U2EwMVYzZTRvYDBdS1FfT2M0bWAwX0tTX09hNGtgMGJLU19PXzRsYDBiS1RfT140bGAwYktUX09eNGxgMGJLVF9PXjRrYDBjS1VfT100a2AwZEtTX09eNGxgMGJLVF9PXjRsYDBiS1RfT140bGAwYktUX09eNGxgMGNLU19PXTRsYDBkS1NfT100bWAwY0tTX09dNG1gMGNLU19PXTRtYDBjS1NfT100bWAwY0tTX09dNG1gMGJLU19PXzRsYDBiS1RfT140bGAwYktUX09eNGxgMGJLVF9PXjRsYDBiS1RfT180a2AwYUtUX09gNGxgMF9LVV9PYTRqYDBgS1ZfT2A0amAwYEtWX09gNGpgMGBLVl9PYDRqYDBgS1VfT2E0a2AwX0tVX09hNGtgMF5LVl9PYjRpYDBfS1dfT2E0aWAwX0tXX09hNGlgMF9LVl9PYjRqYDBeS1ZfT2I0amAwXktWX09iNGpgMF5LVl9PYzRoYDBdS1hfT2Q0aGAwXEtYX09kNGhgMFxLWF9PZDRoYDBcS1hfT2Q0aGAwXEtYX09kNGhgMFxLV19PZTRoYDBbS1lfT2U0Z2AwW0tZX09lNGdgMFtLWV9PZTRnYDBbS1lfT2U0Z2AwW0tYX09mNGhgMFpLWF9PZjRnYDBaS1pfT2Y0ZmAwWktaX09nNGVgMFlLW19PZzRlYDBZS1pfT2g0ZmAwWEtaX09oNGZgMFhLWl9PaDRlYDBZS1tfT2c0ZWAwWEtcX09oNGRgMFhLW19PaTRlYDBXS1tfT2k0ZWAwV0tbX09pNGVgMFdLW19PaTRkYDBYS1xfT2g0ZGAwV0tcX09qNGRgMFZLXF9PajRkYDBWS1xfT2s0Y2AwVUtdX09rNGNgMFVLXV9PazRiYDBWS11fT2s0Y2AwVEteX09sNGJgMFRLXl9PbDRiYDBUS15fT2w0YmAwVEteX09sNGJgMFNLXl9PbjRhYDBRS2FfT280X2AwUEtiX09QNV5gMG1KZV9PUzVbYDBpSmlfT1c1YWMwMTAxTzAwMDAwTzEwMDAwMDAwME8xMDAwMDAwMDBPMk8wUU1iSlhfT141ZmAwZEpaX09cNWRgMGZKXF9PWjViYDBoSl1fT1k1YWAwaEpgX09YNV5gMGpKYl9PVjVdYDBrSmNfT1U1XGAwbEpkX09UNVpgMG5KZV9PUzVaYDBuSmZfT1I1WWAwb0pnX09RNVlgMG5KaF9PUjVYYDBuSmhfT1M1V2AwbUpoX09UNVhgMGxKaF9PVDVYYDBsSmhfT1Q1WGAwbEpoX09UNVhgMGxKZ19PVTVZYDBrSmdfT1U1WWAwakpoX09WNVhgMGpKaF9PVjVYYDBqSmhfT1Y1WGAwakpnX09XNVpgMGhKZl9PWDVaYDBoSmZfT1g1WmAwaEpmX09YNVtgMGZKZl9PWzVZYDBlSmZfT1w1WmAwZEpmX09cNVtgMGNKZV9PXTVbYDBjSmVfT101W2AwY0plX09dNVtgMGNKZF9PXjVcYDBhSmVfT181W2AwYUplX09fNVtgMGFKZV9PXzVbYDBhSmVfT181W2AwYUpkX09gNVxgMGBKZF9PYDVcYDBgSmRfT2A1XGAwX0plX09iNVlgMF9KZ19PYTVZYDBfSmZfT2I1WWAwX0pnX09hNVlgMF9KZ19PYTVYYDBgSmhfT2A1VmAwYkpqX09eNVVgMGNKal9PXjVUYDBjSm1fT101UmAwZEpuX09cNVJgMGRKbl9PXDVSYDBkSm5fT1w1UmAwZEptX09dNVNgMGNKbV9PXTVTYDBjSm1fT141UmAwYUpvX09fNVFgMGFKb19PXzVRYDBhSm5fT2A1UmAwYEpuX09gNVNgMF5Kbl9PYjVSYDBdSm9fT2M1UmAwW0pvX09lNVJgMFlKbl9PZzVTYDBYSm5fT2g1U2AwVkpuX09pNVRgMFVKbV9PazVTYDBUSm5fT2s1VGAwU0psX09uNVVjME8xMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzAxME8xMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTAwTzAxME8xMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzFPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTAwTzAxME8xMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTAwTzAwMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTAwTzAxME8xMDBPMDEwTzEwME8xME8wMTAwTzEwME8wMTBPMTAwTzEwTzAxMDBPMTAwTzAxME8xMDBPMTBPMDEwME8xMDBPMDEwTzEwME8xME8wMTAwTzEwYltPUkxmYDBtM1pfT1RMZWAwbTNbX09TTGVgMGwzW19PVUxlYDBrM1pfT1ZMZmAwaTNaX09YTGZgMGgzWl9PWExlYDBoM1tfT1lMZWAwZzNaX09aTGZgMGUzW19PW0xlYDBlM1pfT11MZWAwYjNbX09fTGRgMGIzXF9PXkxkYDBhM1xfT2BMZGAwYDNbX09hTGVgMF4zW19PY0xlYDBdM1tfT2NMZGAwXTNcX09kTGRgMFwzW19PZUxlYDBaM1xfT2ZMZGAwWjNbX09nTGVgMFgzW19PaUxkYDBXM1xfT2pMZGAwVjNcX09qTGRgMFUzXF9PbExkYDBUM1tfT21MZWAwUjNcX09uTGNgMFMzXF9PbkxkYDBRM1xfT1BNZGAwUDNbX09STWRgMG0yXV9PU01jYDBtMlxfT1RNY2AwbDJdX09VTWNgMGsyXV9PVU1jYDBqMl1fT1dNY2AwaTJcX09YTWNgMGgyXl9PWE1iYDBoMl1fT1lNY2AwZjJdX09bTWNgMGUyXF9PXE1kYDBjMl1fT11NYmAwZDJdX09dTWNgMGIyXV9PX01jYDBhMl1fT19NY2AwYDJdX09hTWNgMF8yXF9PYk1jYDBeMl1fT2NNY2AwXTJdX09jTWNgMFwyXV9PZU1jYDBbMlxfT2dNY2AwWDJeX09oTWFgMFkyXl9PaE1iYDBXMl5fT2pNYmAwVjJdX09rTWNgMFQyXl9PbE1iYDBUMl1fT21NYmAwUzJeX09uTWJgMFIyXl9Pbk1iYDBRMl5fT1BOYmAwUDJdX09RTmNgMG4xXl9PUk5hYDBvMV5fT1JOYmAwbTFeX09UTmJgMGwxXV9PVU5jYDBqMV5fT1ZOYmAwajFdX09XTmJgMGkxXl9PWE5iYDBoMV1fT1lOY2AwZjFeX09bTmFgMGUxXl9PXE5iYDBjMV5fT15OYWAwYzFeX09eTmJgMGExXl9PYE5iYDBgMV5fT2BOYmAwXzFeX09iTmFgMF8xXl9PYk5iYDBdMV5fT2ROYmAwXDFeX09kTmJgMFsxXl9PZk5iYDBaMV1fT2dOYmAwWTFeX09oTmJgMFgxXV9PaU5jYDBWMV5fT2pOYmAwVjFdX09rTmJgMFUxXl9PbE5iYDBUMV1fT21OY2AwUjFdX09vTmNgMFExXV9PUE9iYDBvMF5fT1JPYWAwbzBeX09ST2JgMG0wXl9PVE9iYDBsMF5fT1RPYmAwazBeX09WT2FgMGswXl9PVk9iYDBpMF5fT1hPYmAwZzBeX09aT2JgMGYwXl9PWk9hYDBmMF9fT1tPYWAwZTBeX09cT2JgMGMwXl9PXk9iYDBiMF1fT19PY2AwYDBeX09AYWAwYTBeX09AYmAwP15fT0JiYDA+XV9PQ2NgMDxdX09FYmAwPF5fT0RiYDA7Xl9PR2FgMDleX09IYmAwN15fT0piYDA2Xl9PSmFgMDZfX09LYWAwNV5fT0xiYDAzXl9PTmJgMDJdX09PYmAwMV9fT09hYDAxXl9PMGJgME9eX08yYmAwTl1fTzNjYDBMXV9PNWJgMExeX080YmAwS15fTzZiYDBKXV9PN2NgMEhdX085YmAwSF5fTzhiYDBHXl9POmJgMEZdX088YmAwQ15fTz5iYDBCXV9PP2JgMEFfX08/YWAwQV5fT2AwYmAwX09eX09iMGJgMF5PXV9PYzBiYDBdT15fT2QwYmAwXE9eX09kMGJgMFtPXl9PZjBiYDBaT11fT2cwYmAwWU9eX09oMGJgMFhPXV9PaTBjYDBWT15fT2owYmAwVk9dX09rMGNgMFRPXV9PbTBiYDBUT11fT20wY2AwUk9eX09uMGJgMFJPXV9PbzBjYDBQT11fT1IxYWAwb05eX09SMWJgMG1OXl9PVDFiYDBsTl5fT1QxYmAwa05eX09WMWJgMGpOXV9PVzFiYDBpTl5fT1gxYmAwaE5dX09ZMWNgMGZOXl9PWjFiYDBmTl1fT1sxYmAwZU5eX09cMWJgMGROXV9PXTFjYDBiTl5fT14xYmAwYk5dX09fMWNgMGBOXV9PYTFiYDBgTl1fT2ExY2AwXk5dX09jMWNgMF1OXV9PYzFjYDBcTl1fT2UxYmAwXE5dX09lMWNgMFpOXV9PaDFiYDBYTl1fT2kxY2AwVk5eX09qMWJgMFVOXl9PbDFhYDBVTl5fT2wxYmAwU05eX09uMWJgMFJOXl9PbjFiYDBRTl5fT1AyYWAwUU5eX09QMmJgMG9NXl9PUjJiYDBuTV1fT1MyY2AwbE1eX09UMmFgMG1NXl9PVDJiYDBrTV5fT1YyYmAwak1dX09XMmNgMGhNXV9PWTJjYDBnTV1fT1kyYmAwZ01eX09aMmJgMGZNXV9PWzJjYDBkTV1fT14yYmAwYk1eX09eMmFgMGJNX19PXzJhYDBhTV5fT2AyYmAwX01eX09iMmJgMF5NXV9PYzJjYDBcTV5fT2QyYWAwXU1eX09kMmJgMFtNXl9PZjJiYDBaTV1fT2cyY2AwWE1dX09pMmJgMFhNXl9PaDJiYDBXTV5fT2oyYmAwVk1dX09rMmNgMFRNXV9PbTJjYDBTTV1fT20yYmAwU01eX09uMmJgMFJNXV9PbzJjYDBQTV1fT1EzY2Awb0xcX09TM2JgMG1MX19PUzNhYDBtTF5fT1QzYmAwa0xeX09WM2JgMGpMXV9PVzNjYDBoTF1fT1kzYmAwaExeX09YM2JgMGdMXl9PWjNiYDBmTF1fT1szY2AwZExdX09dM2JgMGRMXl9PXDNiYDBjTF5fT14zYmAwYkxdX09fM2NgMGBMXV9PYTNiYDBgTF1fT2EzY2AwXkxeX09iM2JgMF5MXV9PYzNjYDBcTF1fT2UzY2AwW0xcX09mM2NgMFpMXl9PZjNiYDBaTF1fT2gzYmAwV0xeX09qM2JgMFVMXl9PbDNhYDBVTF5fT2wzYmAwU0xfX09tM2FgMFNMXl9PbjNiYDBRTF5fT1A0YmAwUExdX09RNGJgMG9LXl9PUjRiYDBuS15fT1I0YmAwbUteX09UNGJgMGxLXV9PVTRiYDBrS15fT1Y0YmAwakteX09WNGJgMGlLXl9PWDRiYDBoS11fT1k0Y2AwZktdX09bNGJgMGZLXV9PWzRjYDBkS15fT1w0YmAwZEtdX09eNGJgMGFLXl9PYDRiYDBgS11fT2E0YmAwX0teX09iNGJgMF5LXl9PYjRiYDBdS15fT2Q0YmAwXEtdX09lNGJgMFtLXl9PZjRiYDBaS15fT2Y0YmAwWUteX09oNGJgMFhLXV9PaTRjYDBWS11fT2s0YmAwVktdX09rNGNgMFRLXl9PbDRiYDBUS11fT200Y2AwUktdX09vNGNgMFFLXF9PUDVjYDBQS11fT1E1Y2Awb0pdX09RNWNgMG5KXV9PVDViYDBsSl1fT1U1YmAwa0peX09WNWJgMGpKXl9PVjViYDBpSl5fT1g1YmAwaEpdX09ZNWNgMGZKXV9PWzViYDBmSl1fT1s1Y2AwZEpeX09cNWJgMGRKXV9PXTVjYDBiSl1fT181Y2AwYUpcX09gNWNgMGBKXl9PYDViYDBgSl1fT2E1Y2AwXkpdX09jNWNgMF1KXF9PZDVjYDBcSl5fT2Q1YmAwXEpdX09lNWNgMFpKXV9PZzVjYDBZSlxfT2g1ZGAwV0pcX09rNWJgMFVKX19PazVhYDBVSl5fT2w1YmAwU0peX09uNWJgMFJKXV9PbzVjYDBQSl5fT1A2YWAwUUpeX09QNmJgMG9JXl9PUjZiYDBtSV5fT1Q2YmAwbEleX09UNmFgMGxJX19PVTZhYDBrSV5fT1Y2YmAwaUleX09YNmJgMGhJXl9PWDZiYDBnSV5fT1o2YWAwZ0leX09aNmJgMGVJXl9PXDZiYDBjSV5fT142YmAwYkleX09eNmJgMGFJXl9PYDZhYDBhSV5fT2A2YmAwX0leX09iNmJgMF5JXl9PYjZiYDBdSV5fT2Q2YWAwXElfX09lNmFgMFtJXl9PZjZiYDBZSV9fT2c2YWAwWUleX09oNmJgMFdJXl9PajZhYDBXSV5fT2o2YmAwVUleX09sNmJgMFNJX19PbTZhYDBTSV5fT242YWAwUklfX09vNmFgMFFJXl9PUDdiYDBvSF9fT1E3YWAwb0heX09SN2FgMG5IX19PUzdhYDBtSF5fT1Q3YmAwa0hfX09VN2FgMGpIX19PVzdgYDBqSF9fT1c3YWAwaEhfX09ZN2FgMGdIXl9PWjdiYDBlSF9fT1s3YGAwZkhfX09bN2FgMGRIX19PXTdhYDBjSF5fT143YmAwYUhfX09fN2BgMGJIX19PXzdhYDBgSF9fT2E3YWAwX0heX09iN2FgMF5IYF9PYjdgYDBeSF9fT2M3YWAwXEhfX09lN2FgMFtIXl9PZjdhYDBaSGBfT2Y3YGAwWkhfX09nN2FgMFhIX19PaTdhYDBXSF5fT2o3YWAwVkhfX09rN2FgMFVIX19PazdhYDBUSF9fT203YWAwU0heX09uN2FgMFJIX19PbzdhYDBRSF9fT1A4YGAwb0dgX09SOGBgMG5HX19PUzhgYDBtR2BfT1Q4YGAwbEdgX09UOGBgMGtHYF9PVjhgYDBqR19fT1c4YGAwaUdgX09YOGBgMGhHX19PWThhYDBmR19fT1s4YGAwZkdfX09bOGFgMGRHX19PXThgYDBkR19fT104YWAwYkdfX09fOGBgMGJHYF9PXjhgYDBhR2BfT2A4YGAwYEdfX09hOGBgMF9HYF9PYjhgYDBeR19fT2M4YGAwXUdgX09kOGBgMFxHX19PZThgYDBbR2BfT2Y4YGAwWkdfX09nOGFgMFhHX19PaThhYDBWR19fT2s4YWAwVUdeX09sOGJgMFNHXl9PbjhiYDBRR15fT1A5YmAwb0ZfX09ROWFgMG9GXl9PUjliYDBtRl5fT1Q5YmAwa0ZeX09WOWJgMGpGXV9PVzliYDBpRl5fT1g5YmAwZ0ZeX09aOWFgMGdGXl9PWjlhYDBmRl9fT1s5YWAwZEZfX09dOWBgMGNGYF9PXjlgYDBiRl9fT185YWAwYEZfX09hOWFgMF1GYF9PZDlgYDBZRmNfT2c5XWAwVUZmX09sOVpgMFFGaF9PUDpVYTAxTzFPMU8xTzFQMVBPMk4yTjFPMk40TDNNMk4xTzFPMk4yTjAwMDAwMDAwTzEwME8xTjJNM00zTjIwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAxTzAwMDAwMDAwMDAwMDAwMDAxTzAwMDAwMDFPMDAwMDAwMDAwMDAwMDAwMDAwMDAwMU8wMDAwMDAwMDAwMDAwMU8wMDFPMDAxTzRMMDAwMDAwMDAwMDAwMDAwME8xMDBPMU8xMDBJN08xTjJPMU8xTTNvTm9Gal5PVjlUYTBWR11eT284YmEwajBPMU4yTDRNMzAwMU8wMDFPMDAxTzAwMU8xTzAwMU8wMDFPMDAxTzAwMU8xTzAwMU8wMDFPMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDFPMU8xTzFPMU8xTzJONEwxTzAwMU8wMDAwMDAwMDFPMDBIODAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAxTzAwMDAxTzAwMU8wMDFPMDAxTzAwMDAxTzAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPME8yTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDBPMk8wMDFPMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPME8yTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDBPMk8wMDFPMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPME8yTzAwMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzBPMTAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wTzEwMU8wMDFPMDAwMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDBPMk8wMDFPMDAwMDFPMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwTzJPMDAxTzAwMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFOMTAxTzAwMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTjEwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMDAxTzBPMk8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwTzJPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDBPMk8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTjEwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFOMTAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wTzEwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFOMTAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wTzJPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU4xMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwTzJPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU4xMDFPMDAwMDFPMDAxTzAwMU8wMDAwMU8wMDFPMDAwMDFPMDAxTjEwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAwMDFPMDAwMDFPME8yTzAwMDAxTzAwMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAwMDFPME8xMDFPMDAxTzAwMDAxTzAwMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAwMDFPMDAwMDFOMTAxTzAwMDAxTzAwMDAxTzAwMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDFPMDAwMDFPMDAwMDFPMDAxTzAwMDAxTzBPMTAxTzAwMDAxTzAwMU8wMDAwMU8wMDAwMU8wMDAwMU8wMDFOMTAwTzJPME9YV08=" # noqa: E501
#     mask = dict(counts=s, size=(900, 1600))
#     target = mask_decode(mask)
#     from matplotlib import pyplot as plt
#     plt.imshow(target)
#     plt.show()
