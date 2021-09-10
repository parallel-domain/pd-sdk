import time
from datetime import datetime

from paralleldomain import Scene
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


class TestSensorFrame:
    def test_date_time_type(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.sensor_names
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        assert isinstance(sensor_frame.date_time, datetime)

    def test_lazy_cloud_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.sensor_names
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        assert cloud is not None
        xyz = cloud.xyz
        assert xyz is not None
        assert xyz.shape[0] > 0

    def test_lazy_cloud_caching(self, decoder: DatasetDecoder):
        LAZY_LOAD_CACHE.clear()
        dataset = decoder.get_dataset()
        scene = dataset.get_scene(scene_name=list(dataset.scene_names)[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensor_frame = next(iter(frame.lidar_frames))
        cloud = sensor_frame.point_cloud
        assert cloud is not None
        start = time.time()
        xyz = cloud.xyz
        time1 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0

        scene = dataset.get_scene(scene_name=list(dataset.scene_names)[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensor_frame = next(iter(frame.lidar_frames))
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time2 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time2 < time1
        assert time2 < 1

        scene = dataset.get_scene(scene_name=list(dataset.scene_names)[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensor_frame = next(iter(frame.lidar_frames))
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time3 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time3 < time1
        assert time3 < 1

        scene = dataset.get_scene(scene_name=list(dataset.scene_names)[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensor_frame = next(iter(frame.lidar_frames))
        cloud = sensor_frame.point_cloud
        start = time.time()
        xyz = cloud.xyz
        time4 = time.time() - start
        assert xyz is not None
        assert xyz.shape[0] > 0
        assert time4 < time1
        assert time3 < 1

    def test_lazy_image_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensor_frame = next(iter(frame.camera_frames))
        image = sensor_frame.image
        assert image is not None
        assert isinstance(image.height, int)
        assert isinstance(image.width, int)
        assert isinstance(image.channels, int)
        rgb = image.rgb
        assert rgb is not None
        assert len(rgb.shape) == 3
        assert rgb.shape[0] == image.height
        assert rgb.shape[1] == image.width

    def test_lidar_sensor_frame_ids_are_loaded(self, scene: Scene):
        lidar_name = scene.lidar_names[0]
        lidar_sensor = scene.get_lidar_sensor(sensor_name=lidar_name)
        frame_ids = lidar_sensor.frame_ids
        assert len(frame_ids) > 0
        assert len(scene.frame_ids) >= len(frame_ids)

        for frame_ids in list(frame_ids)[::3]:
            sensor_frame = lidar_sensor.get_frame(frame_id=frame_ids)
            assert sensor_frame.point_cloud is not None
            assert sensor_frame.point_cloud.xyz is not None
            assert sensor_frame.point_cloud.xyz.size > 0

    def test_camera_sensor_frame_ids_are_loaded(self, scene: Scene):
        cam_name = scene.camera_names[0]
        cam_sensor = scene.get_camera_sensor(sensor_name=cam_name)
        frame_ids = cam_sensor.frame_ids
        assert len(frame_ids) > 0
        assert len(scene.frame_ids) >= len(frame_ids)

        for frame_ids in list(frame_ids)[::3]:
            sensor_frame = cam_sensor.get_frame(frame_id=frame_ids)
            assert sensor_frame.image is not None
            assert sensor_frame.image.rgb is not None
            assert sensor_frame.image.rgb.size > 0
