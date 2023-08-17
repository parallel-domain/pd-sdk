from datetime import datetime
from typing import List

import numpy as np

from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import (
    SemanticSegmentation2D,
    InstanceSegmentation2D,
    AnnotationIdentifier,
    BoundingBox2D,
)
from paralleldomain.model.class_mapping import ClassMap, ClassDetail
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame, SensorIntrinsic, SensorExtrinsic, SensorPose
from paralleldomain.model.statistics.aggregated_2d_semantic_segmentation_pxl_counts import (
    Aggregated2DSemanticSegmentationPixelCounts,
)


def get_sem_seg_masks(frame_id: List[str]):
    if frame_id == ["1"]:
        class_ids = np.zeros([512, 512, 1], dtype=int)
        class_ids[0:20, 0:10] = 1
        class_ids[100:120, 100:110] = 2
        class_ids[200:350, 200:400] = 2
        class_ids[200:201, 400:405] = 3
    else:
        class_ids = np.zeros([512, 512, 1], dtype=int)

    return class_ids


def test_recorder(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    SS2D = Aggregated2DSemanticSegmentationPixelCounts()
    for sensor_frame, _, scene in dataset.sensor_frame_pipeline(
        shuffle=True, only_cameras=True, sensor_names=["camera_front"], frame_ids=["0"]
    ):
        SS2D.update(scene=scene, sensor_frame=sensor_frame)

    assert len(SS2D._recorder["semseg_2d_annotations"]) > 0


def test_semantic_segmentation():
    SS2D = Aggregated2DSemanticSegmentationPixelCounts()

    for frame_id in [["1"], ["2"]]:
        scene_decoder = InMemorySceneDecoder(frame_ids=frame_id)
        class_ids = get_sem_seg_masks(frame_id=frame_id)
        scene = Scene(name="test_scene", decoder=scene_decoder)

        image = np.zeros([512, 512, 3], dtype=np.uint8)
        semseg2d = SemanticSegmentation2D(class_ids=class_ids)

        class_map = ClassMap(
            classes=[
                ClassDetail(name="void", id=0, instanced=False),
                ClassDetail(name="pedestrian", id=1, instanced=True),
                ClassDetail(name="car", id=2, instanced=True),
                ClassDetail(name="animal", id=3, instanced=True),
            ]
        )

        semseg_annotation_key = AnnotationIdentifier(annotation_type=SemanticSegmentation2D)
        frame_decoder = InMemoryCameraFrameDecoder(
            dataset_name="test",
            scene_name="test_scene",
            extrinsic=SensorExtrinsic(),
            sensor_pose=SensorPose(),
            annotations={semseg_annotation_key: semseg2d},
            class_maps={semseg_annotation_key: class_map},
            intrinsic=SensorIntrinsic(),
            rgba=image,
            image_dimensions=image.shape,
            distortion_lookup=None,
            metadata={},
            date_time=datetime.now(),
        )

        sensor_frame = CameraSensorFrame(sensor_name="test_sensor", frame_id=frame_id, decoder=frame_decoder)
        SS2D.update(scene=scene, sensor_frame=sensor_frame)

    assert len(SS2D._recorder["semseg_2d_annotations"]) == 2
    assert (
        SS2D._recorder["semseg_2d_annotations"][1]["semseg_2d_pixel_counts"]["void"] == image.shape[0] * image.shape[1]
    )
