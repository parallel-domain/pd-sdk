from datetime import datetime

import numpy as np

from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, BoundingBox2D, BoundingBoxes2D
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.statistics import ClassHeatMaps


def test_bounding_boxes():
    heatmaps = ClassHeatMaps()

    scene_decoder = InMemorySceneDecoder(frame_ids=["1"], scene_name="scene_01")
    scene = Scene(decoder=scene_decoder)

    image = np.zeros([450, 450, 3], dtype=np.uint8)
    boxes = []
    boxes.append(BoundingBox2D(x=0, y=10, width=10, height=20, class_id=1, instance_id=1))
    boxes.append(BoundingBox2D(x=100, y=100, width=10, height=20, class_id=2, instance_id=2))
    boxes.append(BoundingBox2D(x=200, y=200, width=200, height=150, class_id=2, instance_id=3))
    boxes.append(BoundingBox2D(x=300, y=275, width=45, height=32, class_id=2, instance_id=4))
    boxes.append(BoundingBox2D(x=400, y=200, width=5, height=1, class_id=3, instance_id=5))
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
        sensor_name="test_sensor",
        frame_id="1",
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

    sensor_frame = CameraSensorFrame(decoder=frame_decoder)
    heatmaps.update(scene=scene, sensor_frame=sensor_frame)

    heat_map_ped = np.zeros([450, 450], dtype=np.int32)
    heat_map_car = np.zeros([450, 450], dtype=np.int32)
    heat_map_animal = np.zeros([450, 450], dtype=np.int32)
    heat_map_ped[10:30, 0:10] += 1
    heat_map_car[100:120, 100:110] += 1
    heat_map_car[200:350, 200:400] += 1
    heat_map_car[275:307, 300:345] += 1
    heat_map_animal[200:201, 400:405] += 1

    heatmaps = heatmaps.get_heatmaps()

    assert np.all(np.equal(heatmaps["pedestrian"], heat_map_ped))
    assert np.all(np.equal(heatmaps["car"], heat_map_car))
    assert np.all(np.equal(heatmaps["animal"], heat_map_animal))
