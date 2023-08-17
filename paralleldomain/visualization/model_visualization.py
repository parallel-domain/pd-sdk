from collections import deque
from time import sleep

import cv2
from itertools import islice
from typing import List, Union, Optional

import numpy as np
import rerun as rr

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.visualization.initialization import initialize_viewer


def show_dataset(
    dataset: Dataset,
    annotations_to_show: List[AnnotationType] = None,
    max_frames: int = None,
    max_scenes: int = None,
    entity_root: str = "world",
    **kwrags,
):
    scene_stream = dataset.scene_pipeline(**kwrags)
    if max_frames is not None:
        scene_stream = islice(scene_stream, max_scenes)
    for scene in scene_stream:
        rr.reset_time()
        rr.log_cleared(entity_path=entity_root, recursive=True)
        show_scene(scene=scene, annotations_to_show=annotations_to_show, max_frames=max_frames, entity_root=entity_root)


def show_sensor_frame(
    sensor_frame: SensorFrame,
    annotations_to_show: List[AnnotationType] = None,
    depth_cutoff: float = 3000.0,
    entity_root: str = "world",
):
    initialize_viewer(entity_root=entity_root, timeless=False)

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
            elif annotation_type is AnnotationTypes.InstanceSegmentation2D:
                instance_ids = sensor_frame.get_annotations(
                    annotation_type=AnnotationTypes.InstanceSegmentation2D
                ).instance_ids
                rr.log_segmentation_image(entity_path=f"{image_ref}/instance_segmentation_2d", image=instance_ids)
            elif annotation_type is AnnotationTypes.BoundingBoxes3D:
                boxes = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes
                class_map = sensor_frame.class_maps[AnnotationTypes.BoundingBoxes3D]
                for box in boxes:
                    global_pose = box.pose
                    box_meta = dict(instance_id=box.instance_id, num_points=box.num_points)
                    box_meta.update(box.attributes)
                    rr.log_obb(
                        f"{frame_ref}/bounding_boxes_3d/{box.instance_id}",
                        label=class_map[box.class_id].name,
                        class_id=box.class_id,
                        ext=box_meta,
                        half_size=[box.length / 2, box.width / 2, box.height / 2],
                        position=global_pose.translation,
                        rotation_q=[
                            global_pose.quaternion.x,
                            global_pose.quaternion.y,
                            global_pose.quaternion.z,
                            global_pose.quaternion.w,
                        ],
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
                    box_meta = dict(instance_id=box.instance_id, num_points=box.num_points)
                    box_meta.update(box.attributes)
                    rr.log_obb(
                        f"{frame_ref}/bounding_boxes_3d/{box.instance_id}",
                        label=class_map[box.class_id].name,
                        class_id=box.class_id,
                        ext=box_meta,
                        half_size=[box.length / 2, box.width / 2, box.height / 2],
                        position=global_pose.translation,
                        rotation_q=[
                            global_pose.quaternion.x,
                            global_pose.quaternion.y,
                            global_pose.quaternion.z,
                            global_pose.quaternion.w,
                        ],
                    )
                    forward = box.pose @ np.array([[box.length, 0.0, 0.0, 0.0]]).T
                    # forward = box.pose @ np.array([[1., 0., 0., 1.]]).T
                    rr.log_arrow(
                        entity_path=f"{frame_ref}/bounding_boxes_3d/{box.instance_id}/direction",
                        origin=global_pose.translation,
                        vector=forward.T[0, :3],
                    )

        cloud = sensor_frame.point_cloud
        # note: intensity should be between [0, 1] and of shape (:, 1)
        intensity = cloud.intensity
        intensity = np.clip(intensity, 0.0, 1.0)
        pad_ones = np.ones_like(cloud.intensity)
        intensity_color = np.stack([0.25 * intensity, 0.5 + 0.5 * intensity, pad_ones], axis=-1) * 255
        intensity_color = cv2.cvtColor(intensity_color.astype(np.uint8), cv2.COLOR_HSV2RGB)[:, 0, :]

        rr.log_points(
            entity_path=cloud_ref,
            positions=cloud.xyz,
            class_ids=pcl_class_ids,
            colors=intensity_color,
            identifiers=pcl_instance_ids,
        )


def show_frame(
    frame: Frame,
    frame_index: Optional[int] = None,
    annotations_to_show: List[AnnotationType] = None,
    entity_root: str = "world",
):
    sleep(1)
    initialize_viewer(entity_root=entity_root, timeless=False)

    if frame.date_time is not None:
        rr.set_time_seconds(timeline="time", seconds=frame.date_time.timestamp())
    else:
        try:
            frame_int_id = int(frame.frame_id)
        except ValueError:
            frame_int_id = frame_index
        rr.set_time_sequence(timeline="time", sequence=frame_int_id)
        rr.log_cleared(entity_path=entity_root, recursive=True)

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
    initialize_viewer(entity_root=entity_root, timeless=False)

    frame_stream = scene.frame_pipeline()
    if max_frames is not None:
        frame_stream = islice(frame_stream, max_frames)
    for frame_idx, frame in enumerate(frame_stream):
        show_frame(frame=frame, annotations_to_show=annotations_to_show, entity_root=entity_root, frame_index=frame_idx)
