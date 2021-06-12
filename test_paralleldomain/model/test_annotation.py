import time

import numpy as np
import pytest

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D, BoundingBox3D


class TestSensorFrame:
    def test_box_3d_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = next(iter(frame.lidar_frames))
        boxes = lidar_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)

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
        camera_sensor = next(iter(frame.camera_frames))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)

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

    def test_optical_flow_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[5])
        camera_sensor = next(iter(frame.camera_frames))
        flow = camera_sensor.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)

        assert flow is not None
        image = camera_sensor.image.rgb
        assert flow.vectors.shape[2] == 2
        assert image.shape[:2] == flow.vectors.shape[:2]

    def test_image_coordinates(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[5])
        camera_sensor = next(iter(frame.camera_frames))

        rgb = camera_sensor.image.rgb
        coordinates = camera_sensor.image.coordinates
        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                assert np.all(coordinates[y, x] == np.array([y, x]))

    """
    @pytest.skip
    def test_image_warp(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[5])
        camera_sensor = next(iter(frame.camera_frames))
        flow = camera_sensor.get_annotations(
            annotation_type=AnnotationTypes.OpticalFlow
        )


        rgb = camera_sensor.image.rgb
        next_image = np.zeros_like(rgb)
        coordinates = camera_sensor.image.coordinates
        next_frame_coords = coordinates + flow.vectors

        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                next_coord = next_frame_coords[y, x]
                if 0 <= next_coord[0] < rgb.shape[0] and 0 <= next_coord[1] < rgb.shape[1]:
                    next_image[next_coord[0], next_coord[1], :] = rgb[y, x, :]

        import cv2
        cv2.imshow("window_name", next_image[..., [2,1,0]])
        cv2.waitKey()
    """
