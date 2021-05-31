import pytest
from paralleldomain import Scene
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


class TestSceneFrames:
    def test_lazy_cloud_loading(self, scene: Scene):
        frames = scene.frames
        assert len(frames) > 0
        assert len(frames) == len(scene.frame_ids)

    def test_lazy_frame_id_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        pre_size = LAZY_LOAD_CACHE.currsize
        frame_ids = scene.frame_ids  # counts as one item / one list of size 1
        assert pre_size + 1 == LAZY_LOAD_CACHE.currsize
        assert len(frame_ids) > 0

    def test_lazy_frame_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        frame_id = scene.frame_ids[0]
        pre_size = LAZY_LOAD_CACHE.currsize
        frame = scene.get_frame(frame_id=frame_id)  # frame objects are not cached
        assert pre_size == LAZY_LOAD_CACHE.currsize
        assert frame.frame_id == frame_id


class TestSceneSensors:
    def test_lazy_cloud_loading(self, scene: Scene):
        sensors = scene.sensors
        assert len(sensors) > 0
        assert len(sensors) == len(scene.sensor_names)

    def test_lazy_sensor_name_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        pre_size = LAZY_LOAD_CACHE.currsize
        sensor_names = scene.sensor_names  # counts as one item / one list of size 1
        assert pre_size + 1 == LAZY_LOAD_CACHE.currsize
        assert len(sensor_names) > 0

    def test_lazy_sensor_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        sensor_name = scene.sensor_names[0]
        pre_size = LAZY_LOAD_CACHE.currsize
        sensor = scene.get_sensor(sensor_name=sensor_name)  # sensor objects are not cached
        assert pre_size == LAZY_LOAD_CACHE.currsize
        assert sensor.name == sensor_name
