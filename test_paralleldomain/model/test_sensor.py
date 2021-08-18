import time

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import TemporalDecoder
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


class TestSensorFrame:
    def test_lazy_cloud_loading(self, scene: Scene):
        frame_ids = scene.ordered_frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.sensor_names
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        cloud = sensor_frame.point_cloud
        assert cloud is not None
        xyz = cloud.xyz
        assert xyz is not None
        assert xyz.shape[0] > 0

    def test_lazy_cloud_caching(self, decoder: TemporalDecoder):
        LAZY_LOAD_CACHE.clear()
        dataset = decoder.get_dataset()
        scene = dataset.get_scene(scene_name=list(dataset.scene_names)[0])
        frame_ids = scene.ordered_frame_ids
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
        frame_ids = scene.ordered_frame_ids
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
        frame_ids = scene.ordered_frame_ids
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
        frame_ids = scene.ordered_frame_ids
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
        frame_ids = scene.ordered_frame_ids
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
