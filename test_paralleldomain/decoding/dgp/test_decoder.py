import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap


class TestDecoderWithClassIdMap:
    def test_map_all_to_same_semseg2d(self):
        custom_map = ClassMap(class_id_to_class_name={1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        decoder = DGPDecoder(
            dataset_path=os.environ["DATASET_PATH"], custom_map=custom_map, custom_id_map=custom_id_map
        )
        dataset = Dataset.from_decoder(decoder=decoder)
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        assert np.all(semseg.class_ids == 1337)

    def test_map_all_to_same_bbox2d(self):
        custom_map = ClassMap(class_id_to_class_name={1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        decoder = DGPDecoder(
            dataset_path=os.environ["DATASET_PATH"], custom_map=custom_map, custom_id_map=custom_id_map
        )
        dataset = Dataset.from_decoder(decoder=decoder)
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
        assert all([box.class_id == 1337 for box in boxes.boxes])

    def test_map_all_to_same_semseg3d(self):
        custom_map = ClassMap(class_id_to_class_name={1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        decoder = DGPDecoder(
            dataset_path=os.environ["DATASET_PATH"], custom_map=custom_map, custom_id_map=custom_id_map
        )
        dataset = Dataset.from_decoder(decoder=decoder)
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.lidar_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation3D)
        assert np.all(semseg.class_ids == 1337)

    def test_map_all_to_same_bbox3d(self):
        custom_map = ClassMap(class_id_to_class_name={1337: "All"})
        custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})
        decoder = DGPDecoder(
            dataset_path=os.environ["DATASET_PATH"], custom_map=custom_map, custom_id_map=custom_id_map
        )
        dataset = Dataset.from_decoder(decoder=decoder)
        scene = dataset.get_scene(scene_name=dataset.scene_names[0])
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.lidar_frames))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
        assert all([box.class_id == 1337 for box in boxes.boxes])

    def test_error_on_unmatched_class_map(self):
        with pytest.raises(ValueError):
            custom_map = ClassMap(class_id_to_class_name={2: "All"})
            custom_id_map = ClassIdMap(class_id_to_class_id={1: 2, 3: 4})
            _ = DGPDecoder(dataset_path=os.environ["DATASET_PATH"], custom_map=custom_map, custom_id_map=custom_id_map)

    def test_error_on_missing_custom_class_map(self):
        with pytest.raises(ValueError):
            custom_id_map = ClassIdMap(class_id_to_class_id={i: 5 for i in range(100)})
            _ = DGPDecoder(dataset_path=os.environ["DATASET_PATH"], custom_id_map=custom_id_map)
