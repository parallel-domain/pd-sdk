import time

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox3D


class TestSensorFrame:
    def test_lazy_cloud_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        assert cloud is not None
        xyz = cloud.xyz
        assert xyz is not None
        assert xyz.shape[0] > 0

    def test_lazy_cloud_caching(self, decoder: Decoder):
        dataset = Dataset.from_decoder(decoder=decoder)
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        assert cloud is not None
        start = time.time()
        xyz = cloud.xyz
        time1 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0

        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time2 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time2 < time1
        assert time2 < 1

        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time3 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time3 < time1
        assert time3 < 1

        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time4 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time4 < time1
        assert time3 < 1
