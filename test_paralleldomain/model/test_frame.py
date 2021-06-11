import pytest
from paralleldomain import Scene
from paralleldomain.model.frame import Frame


@pytest.fixture()
def frame(scene: Scene) -> Frame:
    frame_id = scene.frame_ids[0]
    return scene.get_frame(frame_id=frame_id)


class TestSceneFrames:
    def test_frame_camera_names_are_loadable(self, frame: Frame):
        camera_names = frame.camera_names
        assert len(camera_names) > 0

    def test_frame_lidar_names_are_loadable(self, frame: Frame):
        lidar_names = frame.lidar_names
        assert len(lidar_names) > 0

    def test_frame_sensors_names_are_loadable(self, frame: Frame):
        sensor_names = frame.sensor_names
        assert len(sensor_names) > 0

    def test_frame_sensors_are_loadable(self, frame: Frame):
        sensor_names = frame.sensor_names
        assert len(sensor_names) > 0
        sensor_frames = frame.sensor_frames
        assert len(sensor_frames) == len(sensor_names)

    def test_frame_lidars_are_loadable(self, frame: Frame):
        lidar_names = frame.lidar_names
        assert len(lidar_names) > 0
        lidar_frames = frame.lidar_frames
        assert len(lidar_frames) == len(lidar_names)

    def test_frame_cameras_are_loadable(self, frame: Frame):
        camera_names = frame.camera_names
        assert len(camera_names) > 0
        camera_frames = frame.camera_frames
        assert len(camera_frames) == len(camera_names)