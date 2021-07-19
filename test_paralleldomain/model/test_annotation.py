import time
from typing import Any, Dict

import numpy as np
import pytest

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D, BoundingBox3D
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap


class TestSensorFrame:
    def test_access_class_map(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = next(iter(frame.lidar_frames))
        camera_sensor = next(iter(frame.camera_frames))
        assert lidar_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).class_map
        assert lidar_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation3D).class_map
        assert camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).class_map
        assert camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).class_map
        assert camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D).class_map

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
            assert isinstance(boxes.class_map[box.class_id].name, str)

    def test_box_2d_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.BoundingBoxes2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter([f for f in frame.camera_frames if "virtual" not in f.sensor_name]))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)

        assert isinstance(boxes.boxes, list)
        assert len(boxes.boxes) > 0

        for box in boxes.boxes:
            assert isinstance(box, BoundingBox2D)
            assert isinstance(box.x, int)
            assert isinstance(box.y, int)
            assert isinstance(box.width, int)
            assert isinstance(box.height, int)
            assert isinstance(box.attributes, Dict)
            assert isinstance(box.class_id, int)
            assert isinstance(boxes.class_map[box.class_id].name, str)

    def test_instance_seg_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.InstanceSegmentation2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        id_mask = camera_sensor.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)

        assert id_mask is not None
        instance_ids = id_mask.instance_ids
        assert instance_ids.shape[2] == 1
        assert len(instance_ids.shape) == 3

        image = camera_sensor.image.rgb
        assert image.shape[:2] == instance_ids.shape[:2]

    def test_sem_seg_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.SemanticSegmentation2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)

        assert semseg is not None
        class_ids = semseg.class_ids
        assert class_ids.shape[2] == 1
        assert len(class_ids.shape) == 3

        image = camera_sensor.image.rgb
        assert image.shape[:2] == class_ids.shape[:2]

    def test_optical_flow_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        flow = camera_sensor.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)

        assert flow is not None
        image = camera_sensor.image.rgb
        assert flow.vectors.shape[2] == 2
        assert image.shape[:2] == flow.vectors.shape[:2]

    def test_image_coordinates(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))

        rgb = camera_sensor.image.rgb
        coordinates = camera_sensor.image.coordinates
        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                assert np.all(coordinates[y, x] == np.array([y, x]))

    def test_map_all_to_same_semseg2d(self, dataset: Dataset):
        custom_map = ClassMap.from_id_label_dict({1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        semseg.update_classes(class_id_map=custom_id_map, class_label_map=custom_map)
        assert np.all(semseg.class_ids == 1337)

    def test_map_all_to_same_bbox2d(self, dataset: Dataset):
        custom_map = ClassMap.from_id_label_dict({1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
        boxes.update_classes(class_id_map=custom_id_map, class_label_map=custom_map)
        assert all([box.class_id == 1337 for box in boxes.boxes])

    def test_map_all_to_same_semseg3d(self, dataset: Dataset):
        custom_map = ClassMap.from_id_label_dict({1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.lidar_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation3D)
        semseg.update_classes(class_id_map=custom_id_map, class_label_map=custom_map)
        assert np.all(semseg.class_ids == 1337)

    def test_map_all_to_same_bbox3d(self, dataset: Dataset):
        custom_map = ClassMap.from_id_label_dict({1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.lidar_frames))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
        boxes.update_classes(class_id_map=custom_id_map, class_label_map=custom_map)
        assert all([box.class_id == 1337 for box in boxes.boxes])

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
