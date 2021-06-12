import time

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.model.annotation import (
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
)


class TestSensorFrame:
    def test_box_3d_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        sensors = frame.available_sensors
        lidar_sensor = next(iter([s for s in sensors if s.startswith("lidar")]))
        sensor_frame = frame.get_sensor(sensor_name=lidar_sensor)
        boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)

        assert isinstance(boxes.boxes, list)
        assert len(boxes.boxes) > 0

        for box in boxes.boxes:
            assert isinstance(box, BoundingBox3D)
            assert isinstance(box.pose.translation, np.ndarray)
            assert isinstance(box.pose.transformation_matrix, np.ndarray)
            assert isinstance(box.class_id, int)
            assert isinstance(boxes.class_map[box.class_id], str)

    def test_box_2d_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.BoundingBoxes2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[5])
        sensors = frame.available_sensors
        camera_sensor = next(iter([s for s in sensors if s.startswith("cam")]))
        sensor_frame = frame.get_sensor(sensor_name=camera_sensor)
        boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)

        assert isinstance(boxes.boxes, list)
        assert len(boxes.boxes) > 0

        for box in boxes.boxes:
            assert isinstance(box, BoundingBox2D)
            assert isinstance(box.x, int)
            assert isinstance(box.y, int)
            assert isinstance(box.width, int)
            assert isinstance(box.height, int)
            assert isinstance(box.visibility, float)
            assert isinstance(box.class_id, int)
            assert isinstance(boxes.class_map[box.class_id], str)
