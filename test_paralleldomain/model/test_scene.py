import pytest
from paralleldomain import Scene


class TestSceneFrames:
    def test_lazy_cloud_loading(self, scene: Scene):
        frames = scene.frames
        assert len(frames) > 0
        assert len(frames) == len(scene.frame_ids)

    def test_lazy_frame_id_loading(self, scene: Scene):
        assert scene._frame_ids is None
        frame_ids = scene.frame_ids
        assert len(frame_ids) > 0
        assert scene._frame_ids is not None

    def test_lazy_frame_loading(self, scene: Scene):
        frame_id = scene.frame_ids[0]
        assert frame_id not in scene._frames
        frame = scene.get_frame(frame_id=frame_id)
        assert frame_id in scene._frames
        assert frame.frame_id == frame_id


class TestSceneSensors:
    def test_lazy_cloud_loading(self, scene: Scene):
        sensors = scene.sensors
        assert len(sensors) > 0
        assert len(sensors) == len(scene.sensor_names)

    def test_lazy_sensor_name_loading(self, scene: Scene):
        assert scene._sensor_names is None
        sensor_names = scene.sensor_names
        assert len(sensor_names) > 0
        assert scene._sensor_names is not None

    def test_lazy_sensor_loading(self, scene: Scene):
        sensor_name = scene.sensor_names[0]
        assert sensor_name not in scene._sensors
        sensor = scene.get_sensor(sensor_name=sensor_name)
        assert sensor_name in scene._sensors
        assert sensor.name == sensor_name
