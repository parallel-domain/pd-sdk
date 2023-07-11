from collections import deque
from itertools import islice
from typing import List, Union

import numpy as np
import rerun as rr

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.visualization.initialization import initialize_viewer


def show_dataset(dataset: Dataset):
    pass


def show_sensor_frame(
    sensor_frame: SensorFrame,
    annotations_to_show: List[AnnotationType] = None,
    depth_cutoff: float = 3000.0,
    entity_root: str = "world",
):
    initialize_viewer(entity_root=entity_root, timeless=sensor_frame.date_time is None)

    if annotations_to_show is None:
        annotation_types = sensor_frame.available_annotation_types
    else:
        annotation_types = list(set(annotations_to_show).intersection(set(sensor_frame.available_annotation_types)))

    pose = sensor_frame.sensor_to_world
    frame_ref = f"{entity_root}/{sensor_frame.sensor_name}"

    rr.log_transform3d(entity_path=frame_ref, transform=rr.TranslationAndMat3(pose.translation, pose.rotation))
    if isinstance(sensor_frame, CameraSensorFrame):
        rr.log_view_coordinates(entity_path=frame_ref, xyz="RDF")  # X=Right, Y=Down, Z=Forward
        image_ref = f"{frame_ref}/image"

        rr.log_pinhole(
            entity_path=image_ref,
            child_from_parent=sensor_frame.intrinsic.camera_matrix,
            width=sensor_frame.image.width,
            height=sensor_frame.image.height,
        )

        rr.log_image(entity_path=image_ref, image=sensor_frame.image.rgb)

        for annotation_type in annotation_types:
            if annotation_type is AnnotationTypes.Depth:
                depth = sensor_frame.get_annotations(annotation_type=AnnotationTypes.Depth).depth
                depth = np.where(depth > depth_cutoff, -1.0, depth)
                rr.log_depth_image(entity_path=f"{image_ref}/depth", image=depth, meter=1.0)
            elif annotation_type is AnnotationTypes.BoundingBoxes2D:
                boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).boxes
                class_map = sensor_frame.class_maps[AnnotationTypes.BoundingBoxes2D]
                rects, labels, class_ids, instance_ids = list(), list(), list(), list()
                for box in boxes:
                    rects.append([box.x, box.y, box.width, box.height])
                    labels.append(class_map[box.class_id].name)
                    class_ids.append(box.class_id)
                    instance_ids.append(box.instance_id)

                rr.log_rects(
                    entity_path=f"{image_ref}/bounding_boxes_2d",
                    rects=np.array(rects),
                    labels=labels,
                    class_ids=class_ids,
                    identifiers=instance_ids,
                )
            elif annotation_type is AnnotationTypes.SemanticSegmentation2D:
                class_ids = sensor_frame.get_annotations(
                    annotation_type=AnnotationTypes.SemanticSegmentation2D
                ).class_ids
                rr.log_segmentation_image(entity_path=f"{image_ref}/semantic_segmentation_2d", image=class_ids)
            elif annotation_type is AnnotationTypes.BoundingBoxes3D:
                boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes
                class_map = sensor_frame.class_maps[AnnotationTypes.BoundingBoxes3D]
                for box in boxes:
                    global_pose = box.pose
                    rot_q = deque(global_pose.quaternion.elements)
                    rot_q.rotate(1)
                    box_meta = dict(instance_id=box.instance_id, num_points=box.num_points)
                    box_meta.update(box.attributes)
                    rr.log_obb(
                        f"{frame_ref}/bounding_boxes_3d/{box.instance_id}",
                        label=class_map[box.class_id].name,
                        class_id=box.class_id,
                        ext=box_meta,
                        half_size=[box.length / 2, box.width / 2, box.height / 2],
                        position=global_pose.translation,
                        rotation_q=np.array(rot_q),
                    )

    elif isinstance(sensor_frame, LidarSensorFrame):
        cloud_ref = f"{frame_ref}/cloud"
        rr.log_view_coordinates(cloud_ref, xyz="FLU")  # X=Right, Y=Down, Z=Forward

        pcl_class_ids = None
        pcl_instance_ids = None
        for annotation_type in annotation_types:
            if annotation_type is AnnotationTypes.SemanticSegmentation3D:
                pcl_class_ids = sensor_frame.get_annotations(
                    annotation_type=AnnotationTypes.SemanticSegmentation3D
                ).class_ids.flatten()
            elif annotation_type is AnnotationTypes.InstanceSegmentation3D:
                pcl_instance_ids = sensor_frame.get_annotations(
                    annotation_type=AnnotationTypes.InstanceSegmentation3D
                ).instance_ids.flatten()
            elif annotation_type is AnnotationTypes.BoundingBoxes3D:
                boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes
                class_map = sensor_frame.class_maps[AnnotationTypes.BoundingBoxes3D]
                for box in boxes:
                    global_pose = box.pose
                    rot_q = deque(global_pose.quaternion.elements)
                    rot_q.rotate(1)
                    box_meta = dict(instance_id=box.instance_id, num_points=box.num_points)
                    box_meta.update(box.attributes)
                    rr.log_obb(
                        f"{frame_ref}/bounding_boxes_3d/{box.instance_id}",
                        label=class_map[box.class_id].name,
                        class_id=box.class_id,
                        ext=box_meta,
                        half_size=[box.length / 2, box.width / 2, box.height / 2],
                        position=global_pose.translation,
                        rotation_q=np.array(rot_q),
                    )

        cloud = sensor_frame.point_cloud
        rr.log_points(
            entity_path=cloud_ref,
            positions=cloud.xyz,
            class_ids=pcl_class_ids,
            colors=cloud.rgb,
            identifiers=pcl_instance_ids,
        )


def show_frame(
    frame: Frame,
    annotations_to_show: List[AnnotationType] = None,
    entity_root: str = "world",
):
    initialize_viewer(entity_root=entity_root, timeless=frame.date_time is None)

    if frame.date_time is not None:
        rr.set_time_seconds(timeline="time", seconds=frame.date_time.timestamp())

    for sensor_frame in frame.camera_frames:
        show_sensor_frame(sensor_frame=sensor_frame, annotations_to_show=annotations_to_show, entity_root=entity_root)
    for sensor_frame in frame.lidar_frames:
        show_sensor_frame(sensor_frame=sensor_frame, annotations_to_show=annotations_to_show, entity_root=entity_root)


def show_scene(
    scene: Union[Scene, UnorderedScene],
    annotations_to_show: List[AnnotationType] = None,
    max_frames: int = None,
    entity_root: str = "world",
):
    initialize_viewer(entity_root=entity_root, timeless=not isinstance(scene, Scene))

    frame_stream = scene.frame_pipeline()
    if max_frames is not None:
        frame_stream = islice(frame_stream, max_frames)
    for frame in frame_stream:
        show_frame(frame=frame, annotations_to_show=annotations_to_show, entity_root=entity_root)
