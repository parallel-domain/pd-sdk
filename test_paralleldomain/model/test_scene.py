from paralleldomain import Scene
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE


class TestSceneFrames:
    def test_lazy_cloud_loading(self, scene: Scene):
        frames = list(scene.frames)
        assert len(frames) > 0
        assert len(frames) == len(scene.frame_ids)

    def test_lazy_frame_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        frame_id = scene.ordered_frame_ids[0]
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

    def test_lazy_sensor_loading(self, scene: Scene):
        LAZY_LOAD_CACHE.clear()
        sensor_name = scene.sensor_names[0]
        pre_size = LAZY_LOAD_CACHE.currsize
        sensor = scene.get_sensor(sensor_name=sensor_name)
        assert pre_size == LAZY_LOAD_CACHE.currsize  # Sensor objects are not cached Atm!
        assert sensor.name == sensor_name

    def test_access_class_map(self, scene: Scene):
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.SemanticSegmentation3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes2D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert scene.get_class_map(annotation_type=AnnotationTypes.SemanticSegmentation2D)
