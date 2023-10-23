from datetime import datetime
from typing import Dict
from unittest import mock
from unittest.mock import MagicMock

from paralleldomain import Scene
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder as DGPDatasetDecoderV0
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder as DGPDatasetDecoderV1
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities.projection import DistortionLookupTable


def class_maps_do_match(map_a: Dict[AnnotationType, ClassMap], map_b: Dict[AnnotationType, ClassMap]):
    assert set(map_a.keys()) == set(map_b.keys())
    for k, v in map_a.items():
        assert len(v) == len(map_b[k])
        assert len(v.class_ids) == len(map_b[k].class_ids)
        assert len(v.class_names) == len(map_b[k].class_names)
        for cid in v.class_ids:
            assert map_b[k][cid] == v[cid]


class TestSensorFrame:
    def test_can_access_neighboring_sensor_frames(self, scene: Scene):
        for sensor_name in scene.sensor_names:
            sensor = scene.get_sensor(sensor_name=sensor_name)
            # ensure correct order so that we get the first sensor frame
            sensor_fids = [f for f in scene.frame_ids if f in sensor.frame_ids]
            sensor_frame = scene.get_frame(frame_id=sensor_fids[0]).get_sensor(sensor_name=sensor_name)
            sf_type = type(sensor_frame)
            frame_count = 1
            while sensor_frame is not None:
                sensor_frame = sensor_frame.next_sensor_frame
                if sensor_frame is not None:
                    frame_count += 1
                    assert isinstance(sensor_frame, sf_type)
            assert frame_count == len(sensor_fids)

    def test_can_access_neighboring_prev_sensor_frames(self, scene: Scene):
        for sensor_name in scene.sensor_names:
            sensor = scene.get_sensor(sensor_name=sensor_name)
            # ensure correct order so that we get the first sensor frame
            sensor_fids = [f for f in scene.frame_ids if f in sensor.frame_ids]
            sensor_frame = scene.get_frame(frame_id=sensor_fids[-1]).get_sensor(sensor_name=sensor_name)
            sf_type = type(sensor_frame)
            frame_count = 1
            while sensor_frame is not None:
                sensor_frame = sensor_frame.previous_sensor_frame
                if sensor_frame is not None:
                    frame_count += 1
                    assert isinstance(sensor_frame, sf_type)
            assert frame_count == len(sensor_fids)

    def is_same_scene(self, a: Scene, b: Scene):
        assert isinstance(a, type(b))
        assert isinstance(b, type(a))
        assert a.name == b.name
        assert a.frame_ids == b.frame_ids
        assert a.sensor_names == b.sensor_names
        assert a.number_of_sensor_frames == b.number_of_sensor_frames

    def test_can_access_scene_from_sensor_frame(self, scene: Scene):
        for sensor_name in scene.sensor_names:
            sensor = scene.get_sensor(sensor_name=sensor_name)
            sensor_frame = next(iter(sensor.sensor_frames))
            self.is_same_scene(a=scene, b=sensor_frame.scene)

    def test_can_access_scene_from_sensor(self, scene: Scene):
        for sensor_name in scene.sensor_names:
            sensor = scene.get_sensor(sensor_name=sensor_name)
            self.is_same_scene(a=scene, b=sensor.scene)

    def test_can_access_scene_from_frame(self, scene: Scene):
        for frame_id in scene.frame_ids:
            frame = scene.get_frame(frame_id=frame_id)
            self.is_same_scene(a=scene, b=frame.scene)

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
        dataset = no_lookup_decoder.get_dataset()
        if isinstance(decoder, (DGPDatasetDecoderV0, DGPDatasetDecoderV1)):
            calib_path = dataset._decoder.get_path() / dataset.scene_names[0] / "calibration"
            for camera_name in dataset.camera_names:
                sensor_frame = next(iter(dataset.get_sensor_frames(sensor_names=[camera_name])))
                distortion_lookup = sensor_frame.distortion_lookup
                if (calib_path / (camera_name + ".csv")).exists():
                    assert isinstance(distortion_lookup, DistortionLookupTable)
                else:
                    assert distortion_lookup is None
        else:
            sensor_frame = next(iter(dataset.camera_frames))
            distortion_lookup = sensor_frame.distortion_lookup
            assert distortion_lookup is None

        no_lookup_decoder.lazy_load_cache.clear()
        fake_loopup = mock.MagicMock()

        init_kwargs["settings"] = DecoderSettings(distortion_lookups={sensor_frame.sensor_name: fake_loopup})
        with_lookup_decoder = decoder_cls(**init_kwargs)
        dataset = with_lookup_decoder.get_dataset()
        sensor_frame = next(iter(dataset.get_sensor_frames(sensor_names=[sensor_frame.sensor_name])))
        distortion_lookup = sensor_frame.distortion_lookup
        assert distortion_lookup == fake_loopup

    def test_get_available_annotation_types(self):
        decoder = MagicMock()
        decoder.get_available_annotation_identifiers.return_value = [
            AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D, name="someName"),
        ]
        scene = Scene(decoder=decoder)

        result = scene.available_annotation_types

        assert set(result) == {
            AnnotationTypes.BoundingBoxes2D,
            AnnotationTypes.SemanticSegmentation2D,
        }

    def test_get_annotation_identifiers_of_type(self):
        decoder = MagicMock()
        decoder.get_available_annotation_identifiers.return_value = [
            AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D, name="someName"),
        ]
        scene = Scene(decoder=decoder)

        result = scene.get_annotation_identifiers_of_type(annotation_type=AnnotationTypes.BoundingBoxes2D)
        assert result == [AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D)]

        result = scene.get_annotation_identifiers_of_type(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        assert result == [
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D, name="someName"),
        ]


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
