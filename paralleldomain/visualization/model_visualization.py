import datetime
from itertools import islice
from typing import List, Optional, Union

import cv2
import numpy as np
import rerun as rr

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities import clip_with_warning
from paralleldomain.visualization import initialize_viewer

np_uint16_min, np_uint16_max = np.iinfo(np.uint16).min, np.iinfo(np.uint16).max


def show_dataset(
    dataset: Dataset,
    annotations_to_show: List[Union[AnnotationType, AnnotationIdentifier]] = None,
    max_frames: int = None,
    max_scenes: int = None,
    entity_root: str = "world",
    **kwargs,
):
    scene_stream = dataset.scene_pipeline(**kwargs)
    if max_frames is not None:
        scene_stream = islice(scene_stream, max_scenes)
    for scene in scene_stream:
        rr.reset_time()
        rr.log(entity_root, rr.Clear(recursive=True))
        show_scene(scene=scene, annotations_to_show=annotations_to_show, max_frames=max_frames, entity_root=entity_root)


def show_sensor_frame(
    sensor_frame: SensorFrame,
    annotations_to_show: List[Union[AnnotationType, AnnotationIdentifier]] = None,
    depth_cutoff: float = 3000.0,
    entity_root: str = "world",
):
    initialize_viewer(
        application_id=sensor_frame.dataset_name,
        recording_id=f"{sensor_frame.dataset_name}-{sensor_frame.scene_name}",
        entity_root=entity_root,
        timeless=False,
    )

    if annotations_to_show is None:
        annotation_identifiers = sensor_frame.available_annotation_identifiers
    else:
        annotation_identifiers = [
            ai
            for ai in sensor_frame.available_annotation_identifiers
            if ai.annotation_type in annotations_to_show or ai in annotations_to_show
        ]

    pose = sensor_frame.sensor_to_world
    frame_ref = f"{entity_root}/{sensor_frame.sensor_name}"

    rr.log(
        frame_ref,
        rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation),
    )
    if isinstance(sensor_frame, CameraSensorFrame):
        image_ref = f"{frame_ref}/{sensor_frame.sensor_name}"

        rr.log(
            frame_ref,
            rr.ViewCoordinates(xyz=rr.components.ViewCoordinates(coordinates=[3, 2, 5])),  # RDF
        )
        rr.log(
            image_ref,
            rr.Pinhole(
                image_from_camera=sensor_frame.intrinsic.camera_matrix,
                width=sensor_frame.image.width,
                height=sensor_frame.image.height,
            ),
        )
        rr.log(image_ref, rr.Image(data=sensor_frame.image.rgb))

        annotation_ref = image_ref
        for annotation_identifier in annotation_identifiers:
            annotation_type = annotation_identifier.annotation_type
            if annotation_type is AnnotationTypes.Depth:
                depth = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).depth
                depth = np.clip(depth, -depth_cutoff, depth_cutoff)
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.DepthImage(
                        data=depth[..., 0],
                        meter=1,
                    ),
                )
            elif annotation_type is AnnotationTypes.BoundingBoxes2D:
                class_map = sensor_frame.class_maps[annotation_identifier]
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.AnnotationContext(context=class_descriptions_from_class_map(class_map=class_map)),
                )

                boxes = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).boxes
                if not boxes:
                    continue

                sizes, mins, class_ids, instance_ids, metadata = [], [], [], [], dict(visibility=[], truncation=[])
                for box in boxes:
                    sizes.append([box.width, box.height])
                    mins.append([box.x, box.y])
                    instance_ids.append(box.instance_id)
                    class_ids.append(box.class_id)
                    for k, v in metadata.items():
                        v.append(box.attributes.get(k, box.attributes.get("user_data", {}).get(k, None)))

                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.Boxes2D(sizes=sizes, mins=mins, instance_keys=instance_ids, class_ids=class_ids),
                    rr.AnyValues(**metadata),
                )
            elif annotation_type is AnnotationTypes.Points2D:
                class_map = sensor_frame.class_maps[annotation_identifier]
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.AnnotationContext(context=class_descriptions_from_class_map(class_map=class_map)),
                )

                points = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).points
                # filter out-of-bounds points
                image = sensor_frame.image
                points = [p for p in points if 0 <= p.x < image.width and 0 <= p.y < image.height]

                # filter out points that have "visibility" attribute, and it's set to False
                points = [p for p in points if p.attributes.get("visibility", True) is not False]

                if not points:
                    continue

                positions, class_ids, instance_ids = [], [], []
                for point in points:
                    positions.append([point.x, point.y])
                    class_ids.append(point.class_id)
                    instance_ids.append(point.instance_id)

                # this is a workaround for the fact that points don't have unique identifiers currently
                instance_ids = [i for i in range(len(positions))]

                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.Points2D(
                        positions=np.array(positions),
                        class_ids=clip_with_warning(arr=np.asarray(class_ids), dtype=np.uint16),
                        instance_keys=instance_ids,
                    ),
                )

            elif annotation_type is AnnotationTypes.SemanticSegmentation2D:
                class_map = sensor_frame.class_maps[annotation_identifier]
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.AnnotationContext(context=class_descriptions_from_class_map(class_map=class_map)),
                )
                class_ids = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).class_ids
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.SegmentationImage(data=np.squeeze(class_ids) if class_ids.ndim > 2 else class_ids),
                )
            elif annotation_type is AnnotationTypes.InstanceSegmentation2D:
                instance_ids = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).instance_ids
                rr.log(
                    f"{annotation_ref}/{annotation_identifier}",
                    rr.SegmentationImage(data=np.squeeze(instance_ids) if instance_ids.ndim > 2 else instance_ids),
                )
    elif isinstance(sensor_frame, LidarSensorFrame):
        cloud_ref = f"{frame_ref}/{sensor_frame.sensor_name}"
        rr.log_view_coordinates(cloud_ref, xyz="FLU")  # X=Right, Y=Down, Z=Forward

        pcl_class_ids = None
        pcl_instance_ids = None
        for annotation_identifier in annotation_identifiers:
            annotation_type = annotation_identifier.annotation_type
            if annotation_type is AnnotationTypes.SemanticSegmentation3D:
                pcl_class_ids = sensor_frame.get_annotations(
                    annotation_identifier=annotation_identifier
                ).class_ids.flatten()
            elif annotation_type is AnnotationTypes.InstanceSegmentation3D:
                pcl_instance_ids = sensor_frame.get_annotations(
                    annotation_identifier=annotation_identifier
                ).instance_ids.flatten()
            elif annotation_type is AnnotationTypes.BoundingBoxes3D:
                boxes = sensor_frame.get_annotations(annotation_identifier=annotation_identifier).boxes
                class_map = sensor_frame.class_maps[annotation_identifier]
                rr.log_cleared(entity_path=f"{frame_ref}/{annotation_identifier}", recursive=True)
                for box in boxes:
                    global_pose = box.pose
                    box_meta = dict(instance_id=box.instance_id, num_points=box.num_points)
                    box_meta.update(box.attributes)
                    rr.log_obb(
                        f"{frame_ref}/{annotation_identifier}/{box.instance_id}",
                        label=(
                            class_map[box.class_id].name if box.class_id in class_map.class_ids else str(box.class_id)
                        ),
                        class_id=clip_with_warning(arr=box.class_id, dtype=np.uint16),
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
                        entity_path=f"{frame_ref}/{annotation_identifier}/{box.instance_id}/direction",
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
    annotations_to_show: List[Union[AnnotationType, AnnotationIdentifier]] = None,
    entity_root: str = "world",
):
    initialize_viewer(
        application_id=frame.dataset_name,
        recording_id=f"{frame.dataset_name}-{frame.scene_name}",
        entity_root=entity_root,
        timeless=False,
    )

    if frame.date_time is None:
        rr.log(entity_root, rr.Clear(recursive=True))
        try:
            frame_int_id = int(frame.frame_id)
        except ValueError:
            frame_int_id = frame_index
        rr.set_time_sequence(timeline="frame_idx", sequence=frame_int_id)
    else:
        rr.set_time_seconds(timeline="seconds", seconds=frame.date_time.timestamp())

    for sensor_frame in frame.camera_frames:
        show_sensor_frame(sensor_frame=sensor_frame, annotations_to_show=annotations_to_show, entity_root=entity_root)
    for sensor_frame in frame.lidar_frames:
        show_sensor_frame(sensor_frame=sensor_frame, annotations_to_show=annotations_to_show, entity_root=entity_root)


def show_scene(
    scene: Union[Scene, UnorderedScene],
    annotations_to_show: List[Union[AnnotationType, AnnotationIdentifier]] = None,
    max_frames: int = None,
    entity_root: str = "world",
):
    initialize_viewer(
        application_id=scene.dataset_name,
        recording_id=f"{scene.dataset_name}-{scene.name}",
        entity_root=entity_root,
        timeless=False,
    )

    frame_stream = scene.frame_pipeline()
    if max_frames is not None:
        frame_stream = islice(frame_stream, max_frames)
    for frame_idx, frame in enumerate(frame_stream):
        show_frame(frame=frame, annotations_to_show=annotations_to_show, entity_root=entity_root, frame_index=frame_idx)


def class_descriptions_from_class_map(class_map: ClassMap) -> List[rr.AnnotationInfo]:
    class_descriptions = [
        rr.AnnotationInfo(
            id=class_key,
            label=class_detail.name,
            color=(
                class_detail.meta["color"]["r"],
                class_detail.meta["color"]["g"],
                class_detail.meta["color"]["b"],
                255,
            ),
        )
        for class_key, class_detail in class_map.items()
    ]

    return class_descriptions
