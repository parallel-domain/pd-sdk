from datetime import datetime

import numpy as np

from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import (
    BoundingBox2D,
    BoundingBoxes2D,
    SemanticSegmentation2D,
    InstanceSegmentation2D,
    AnnotationIdentifier,
)
from paralleldomain.model.class_mapping import ClassMap, ClassDetail
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame, SensorIntrinsic, SensorExtrinsic, SensorPose
from paralleldomain.model.statistics import ClassDistribution


def test_dataset_distribution(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    source_annotation_types = [SemanticSegmentation2D]
    class_distribution = ClassDistribution(source_annotation_types=source_annotation_types)
    for sensor_frame, _, scene in dataset.sensor_frame_pipeline(
        shuffle=True, only_cameras=True, sensor_names=["camera_front"], frame_ids=["0"]
    ):
        class_distribution.update(scene=scene, sensor_frame=sensor_frame)

    instance_counts = class_distribution.get_instance_distribution(annotation_type=SemanticSegmentation2D)
    assert len(instance_counts.keys()) > 0


def test_bounding_boxes():
    source_annotation_types = [BoundingBoxes2D]
    class_distribution = ClassDistribution(source_annotation_types=source_annotation_types)

    scene_decoder = InMemorySceneDecoder(frame_ids=["1"])
    scene = Scene(name="test_scene", decoder=scene_decoder)

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    boxes = []
    boxes.append(BoundingBox2D(x=0, y=10, width=10, height=20, class_id=1, instance_id=1))
    boxes.append(BoundingBox2D(x=100, y=100, width=10, height=20, class_id=2, instance_id=2))
    boxes.append(BoundingBox2D(x=200, y=200, width=200, height=150, class_id=2, instance_id=3))
    boxes.append(BoundingBox2D(x=400, y=200, width=5, height=1, class_id=3, instance_id=4))
    bbox_2d = BoundingBoxes2D(boxes=boxes)

    bbox_classmap = ClassMap(
        classes=[
            ClassDetail(name="pedestrian", id=1, instanced=True),
            ClassDetail(name="car", id=2, instanced=True),
            ClassDetail(name="animal", id=3, instanced=True),
        ]
    )
    annotation_key = AnnotationIdentifier(annotation_type=BoundingBoxes2D)
    frame_decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test_scene",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={annotation_key: bbox_2d},
        class_maps={annotation_key: bbox_classmap},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )

    sensor_frame = CameraSensorFrame("test_sensor", "1", decoder=frame_decoder)
    class_distribution.update(scene=scene, sensor_frame=sensor_frame)

    TEST_INSTANCE_COUNTS = {"pedestrian": 1, "car": 2, "animal": 1}
    TEST_PIXEL_COUTNS = {"pedestrian": 200, "car": 30200, "animal": 5}

    assert class_distribution.get_instance_distribution(annotation_type=BoundingBoxes2D) == TEST_INSTANCE_COUNTS
    assert class_distribution.get_pixel_distribution(annotation_type=BoundingBoxes2D) == TEST_PIXEL_COUTNS


def test_semantic_segmentation():
    source_annotation_types = [SemanticSegmentation2D]
    class_distribution = ClassDistribution(source_annotation_types=source_annotation_types)

    scene_decoder = InMemorySceneDecoder(frame_ids=["1"])
    scene = Scene(
        name="test_scene",
        decoder=scene_decoder,
    )

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    class_ids = np.zeros([512, 512, 1], dtype=int)
    instance_ids = np.zeros([512, 512, 1], dtype=int)
    class_ids[0:20, 0:10] = 1
    instance_ids[0:20, 0:10] = 1
    class_ids[100:120, 100:110] = 2
    instance_ids[100:120, 100:110] = 2
    class_ids[200:350, 200:400] = 2
    instance_ids[200:350, 200:400] = 3
    class_ids[200:201, 400:405] = 3
    instance_ids[200:201, 400:405] = 4

    semseg2d = SemanticSegmentation2D(class_ids=class_ids)
    instance2d = InstanceSegmentation2D(instance_ids=instance_ids)

    class_map = ClassMap(
        classes=[
            ClassDetail(name="void", id=0, instanced=False),
            ClassDetail(name="pedestrian", id=1, instanced=True),
            ClassDetail(name="car", id=2, instanced=True),
            ClassDetail(name="animal", id=3, instanced=True),
        ]
    )

    semseg_annotation_key = AnnotationIdentifier(annotation_type=SemanticSegmentation2D)
    instance_seg_annotation_key = AnnotationIdentifier(annotation_type=InstanceSegmentation2D)
    frame_decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test_scene",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={semseg_annotation_key: semseg2d, instance_seg_annotation_key: instance2d},
        class_maps={semseg_annotation_key: class_map, instance_seg_annotation_key: class_map},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )

    sensor_frame = CameraSensorFrame("test_sensor", "1", decoder=frame_decoder)
    class_distribution.update(scene=scene, sensor_frame=sensor_frame)

    TEST_INSTANCE_COUNTS = {"pedestrian": 1, "car": 2, "animal": 1}
    TEST_PIXEL_COUNTS = {"pedestrian": 200, "car": 30200, "animal": 5, "void": 231739}

    assert class_distribution.get_instance_distribution(annotation_type=SemanticSegmentation2D) == TEST_INSTANCE_COUNTS
    assert class_distribution.get_pixel_distribution(annotation_type=SemanticSegmentation2D) == TEST_PIXEL_COUNTS
