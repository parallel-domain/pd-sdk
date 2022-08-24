import time
from collections import defaultdict
from datetime import datetime
from typing import Dict
from unittest import mock

from paralleldomain import Scene
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


def class_maps_do_match(map_a: Dict[AnnotationType, ClassMap], map_b: Dict[AnnotationType, ClassMap]):
    assert set(map_a.keys()) == set(map_b.keys())
    for k, v in map_a.items():
        assert len(v) == len(map_b[k])
        assert len(v.class_ids) == len(map_b[k].class_ids)
        assert len(v.class_names) == len(map_b[k].class_names)
        for cid in v.class_ids:
            assert map_b[k][cid] == v[cid]


class TestSensorFrame:
    def test_date_time_type(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = frame.lidar_names[0]
        sensor_frame = frame.get_lidar(lidar_name=lidar_sensor)
        assert isinstance(sensor_frame.date_time, datetime)

    def test_lazy_cloud_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = frame.lidar_names[0]
        sensor_frame = frame.get_lidar(lidar_name=lidar_sensor)
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
        assert time2 <= time1
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
        assert time3 <= time1
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
        assert time4 <= time1
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

    def test_camera_frame_class_map_access(self, scene: Scene):
        if scene.number_of_camera_frames > 0:
            sensor_frame = next(iter(scene.camera_frames))
            class_maps = sensor_frame.class_maps
            scene_class_maps = scene.class_maps
            class_maps_do_match(map_a=class_maps, map_b=scene_class_maps)

    def test_lidar_frame_class_map_access(self, scene: Scene):
        if scene.number_of_lidar_frames > 0:
            sensor_frame = next(iter(scene.lidar_frames))
            class_maps = sensor_frame.class_maps
            scene_class_maps = scene.class_maps
            class_maps_do_match(map_a=class_maps, map_b=scene_class_maps)

    def test_radar_frame_class_map_access(self, scene: Scene):
        if scene.number_of_radar_frames > 0:
            sensor_frame = next(iter(scene.radar_frames))
            class_maps = sensor_frame.class_maps
            scene_class_maps = scene.class_maps
            class_maps_do_match(map_a=class_maps, map_b=scene_class_maps)

    def test_distortion_loop_access(self, decoder: DatasetDecoder):
        decoder_cls = decoder.__class__
        init_kwargs = decoder.get_decoder_init_kwargs()
        init_kwargs["settings"] = None
        no_lookup_decoder = decoder_cls(**init_kwargs)
        datast = no_lookup_decoder.get_dataset()
        sensor_frame = next(iter(datast.camera_frames))
        distortion_lookup = sensor_frame.distortion_lookup
        assert distortion_lookup is None
        no_lookup_decoder.lazy_load_cache.clear()
        fake_loopup = mock.MagicMock()

        init_kwargs["settings"] = DecoderSettings(distortion_lookups={sensor_frame.sensor_name: fake_loopup})
        with_lookup_decoder = decoder_cls(**init_kwargs)
        dataset = with_lookup_decoder.get_dataset()
        sensor_frame = next(iter(dataset.camera_frames))
        distortion_lookup = sensor_frame.distortion_lookup
        assert distortion_lookup == fake_loopup


class TestSensor:
    def test_lidar_sensor_frame_ids_are_loaded(self, scene: Scene):
        lidar_name = scene.lidar_names[0]
        lidar_sensor = scene.get_lidar_sensor(lidar_name=lidar_name)
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
        cam_sensor = scene.get_camera_sensor(camera_name=cam_name)
        frame_ids = cam_sensor.frame_ids
        assert len(frame_ids) > 0
        assert len(scene.frame_ids) >= len(frame_ids)

        for frame_ids in list(frame_ids)[::3]:
            sensor_frame = cam_sensor.get_frame(frame_id=frame_ids)
            assert sensor_frame.image is not None
            assert sensor_frame.image.rgb is not None
            assert sensor_frame.image.rgb.size > 0
