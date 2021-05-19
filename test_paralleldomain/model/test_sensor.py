import numpy as np
from paralleldomain import Scene
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

    def test_box_3d_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBox3D)

        assert isinstance(boxes, list)
        assert len(boxes) > 0

        for box in boxes:
            assert isinstance(box, BoundingBox3D)
            assert isinstance(box.pose.translation, np.ndarray)
            assert isinstance(box.pose.transformation_matrix, np.ndarray)
