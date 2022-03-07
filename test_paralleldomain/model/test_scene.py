from paralleldomain import Scene
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


class TestSceneFrames:
    def test_lazy_cloud_loading(self, scene: Scene):
        frames = scene.frames
        assert len(frames) > 0
        assert len(frames) == len(scene.frame_ids)

    def test_lazy_frame_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        frame_id = scene.frame_ids[0]
        pre_size = LAZY_LOAD_CACHE.currsize
        frame = scene.get_frame(frame_id=frame_id)  # frame objects are not cached
        assert pre_size == LAZY_LOAD_CACHE.currsize
        assert frame.frame_id == frame_id


class TestSceneSensors:
    def test_lazy_sensor_name_loading(self, scene: Scene):
        sensors = list(scene.sensors)
        assert len(sensors) > 0
        assert len(sensors) == len(scene.sensor_names)

    def test_camera_names_loading(self, scene: Scene):
        camera_names = scene.camera_names
        assert len(camera_names) > 0
        assert len(list(scene.cameras)) == len(camera_names)

    def test_lidar_name_loading(self, scene: Scene):
        lidar_names = scene.lidar_names
        assert len(lidar_names) > 0
        assert len(list(scene.lidars)) == len(lidar_names)

    def test_access_class_map(self, scene: Scene):
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.SemanticSegmentation3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes2D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.SemanticSegmentation2D)

    def test_load_all_class_maps(self, scene: Scene):
        class_maps = scene.class_maps
        for annotype in scene.available_annotation_types:
            assert annotype in class_maps
        assert class_maps is not None

    def test_lidar_frame_loading(self, scene: Scene):
        num_frames = 0
        for sensor in scene.lidars:
            num_frames += len(sensor.frame_ids)

        lidar_frames = list(scene.lidar_frames)
        assert len(lidar_frames) == num_frames
        assert len(lidar_frames) == scene.number_of_lidar_frames
        for frame in lidar_frames:
            assert frame is not None
            assert isinstance(frame, LidarSensorFrame)

    def test_camera_frame_loading(self, scene: Scene):
        num_frames = 0
        for sensor in scene.cameras:
            num_frames += len(sensor.frame_ids)

        cameras_frames = list(scene.camera_frames)
        assert len(cameras_frames) == num_frames
        assert len(cameras_frames) == scene.number_of_camera_frames
        for frame in cameras_frames:
            assert frame is not None
            assert isinstance(frame, CameraSensorFrame)

    def test_sensor_frame_loading(self, scene: Scene):
        num_frames = 0
        use_sensor_names = scene.sensor_names[::2]
        use_sensor_frameids = set()
        for sensor_name in use_sensor_names:
            sensor = scene.get_sensor(sensor_name=sensor_name)
            for i, fid in enumerate(sensor.frame_ids):
                if i % 3 == 0:
                    num_frames += 1
                    use_sensor_frameids.add(fid)

        sensor_frames = list(scene.get_sensor_frames(sensor_names=use_sensor_names, frame_ids=use_sensor_frameids))
        assert len(sensor_frames) == num_frames
        for frame in sensor_frames:
            assert frame is not None
            assert isinstance(frame, CameraSensorFrame) or isinstance(frame, LidarSensorFrame)
