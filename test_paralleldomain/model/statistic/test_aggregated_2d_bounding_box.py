from datetime import datetime
from typing import List

import numpy as np

from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import (
    BoundingBox2D,
    BoundingBoxes2D,
    AnnotationIdentifier,
)
from paralleldomain.model.class_mapping import ClassMap, ClassDetail
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame, SensorIntrinsic, SensorExtrinsic, SensorPose
from paralleldomain.model.statistics.aggregated_2d_bounding_box_annotations import Aggregated2DBoundingBoxAnnotations


def get_boxes(frame_id: List[str]):
    if frame_id == ["1"]:
        boxes = []
        boxes.append(BoundingBox2D(x=0, y=10, width=10, height=20, class_id=1, instance_id=1, attributes=[]))
        boxes.append(BoundingBox2D(x=100, y=100, width=10, height=20, class_id=2, instance_id=2, attributes=[]))
        boxes.append(BoundingBox2D(x=200, y=200, width=200, height=150, class_id=2, instance_id=3, attributes=[]))
        boxes.append(BoundingBox2D(x=400, y=200, width=5, height=1, class_id=3, instance_id=4, attributes=[]))
        return BoundingBoxes2D(boxes=boxes)
    else:
        return BoundingBoxes2D(boxes=[])  # no boxes at all


def test_recorder(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    BB2D = Aggregated2DBoundingBoxAnnotations()
    for sensor_frame, _, scene in dataset.sensor_frame_pipeline(
        shuffle=True, only_cameras=True, sensor_names=["camera_front"], frame_ids=["0"]
    ):
        BB2D.update(scene=scene, sensor_frame=sensor_frame)

    assert len(BB2D._recorder["bbox_2d_annotations"]) > 0


def test_aggregation_of_2d_bbox_statistics():
    BB2D = Aggregated2DBoundingBoxAnnotations()

    for frame_id in [["1"], ["2"], ["3"]]:
        scene_decoder = InMemorySceneDecoder(frame_ids=frame_id)
        scene = Scene(name="test_scene", decoder=scene_decoder)

        image = np.zeros([512, 512, 3], dtype=np.uint8)
        # assigns dummy bounding boxes to the first frame, an empty list of boxes to all other frames
        bbox_2d = get_boxes(frame_id=frame_id)

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
            # for frame 1 we have bbox annotations with a couple of boxes drawn (see the fnc get_boxes above)
            # for frame 2 we have bbox annotations, but the image doesn't contain any foreground objects
            # for frame 3 we have no annotations
            annotations={annotation_key: bbox_2d} if frame_id != ["3"] else {},
            class_maps={annotation_key: bbox_classmap} if frame_id != ["3"] else {},
            intrinsic=SensorIntrinsic(),
            rgba=image,
            image_dimensions=image.shape,
            distortion_lookup=None,
            metadata={},
            date_time=datetime.now(),
        )

        sensor_frame = CameraSensorFrame(sensor_name="test_sensor", frame_id=frame_id[0], decoder=frame_decoder)
        BB2D.update(scene=scene, sensor_frame=sensor_frame)

    assert "skipped_frames" in BB2D._recorder
    assert len(BB2D._recorder["bbox_2d_annotations"]) == 2
    assert len(BB2D._recorder["bbox_2d_annotations"][0]["bboxes_2d"]) == 4
    assert len(BB2D._recorder["bbox_2d_annotations"][1]["bboxes_2d"]) == 0
    assert len(BB2D._recorder["skipped_frames"]) == 1
