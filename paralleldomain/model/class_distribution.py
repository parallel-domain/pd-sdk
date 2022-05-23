from collections import Counter
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Union

import numpy as np

from paralleldomain import Dataset
from paralleldomain.model.annotation import (
    AnnotationType,
    AnnotationTypes,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.annotation.bounding_box_2d import BoundingBoxes2D
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import CameraSensor
from paralleldomain.model.type_aliases import SensorName
from paralleldomain.model.unordered_scene import UnorderedScene


@dataclass
class ClassDistributionInfo:
    class_name: str
    class_instance_count: int
    class_instance_percentage: float
    class_pixel_count: int
    class_pixel_percentage: float
    class_bbox_count: int
    class_bbox_percentage: float


class ClassDistribution:
    def __init__(self):
        self._class_distribution_infos_2d: Dict[str, ClassDistributionInfo] = dict()
        self.total_pixel_count: int = 0
        self.total_instances_2d_count: int = 0
        self.total_bbox_count: int = 0

    @property
    def class_distribution_infos(self) -> List[ClassDistributionInfo]:
        return list(self._class_distribution_infos_2d.values())

    def get_class_info(self, class_name: str) -> ClassDistributionInfo:
        return self._class_distribution_infos_2d.setdefault(
            class_name,
            ClassDistributionInfo(
                class_name=class_name,
                class_instance_count=0,
                class_instance_percentage=0.0,
                class_pixel_count=0,
                class_pixel_percentage=0.0,
                class_bbox_count=0,
                class_bbox_percentage=0.0,
            ),
        )

    def add_pixel_count(self, class_name: str, count: int):
        class_info = self.get_class_info(class_name=class_name)
        class_info.class_pixel_count += int(count)
        self.total_pixel_count += int(count)
        if self.total_pixel_count > 0:
            for ci in self.class_distribution_infos:
                ci.class_pixel_percentage = 100 * (ci.class_pixel_count / self.total_pixel_count)

    def add_instance_count(self, class_name: str, count: int):
        class_info = self.get_class_info(class_name=class_name)
        class_info.class_instance_count += int(count)
        self.total_instances_2d_count += int(count)
        if self.total_instances_2d_count > 0:
            for ci in self.class_distribution_infos:
                ci.class_instance_percentage = 100 * (ci.class_instance_count / self.total_instances_2d_count)

    def add_bbox_count(self, class_name: str, count: int):
        class_info = self.get_class_info(class_name=class_name)
        class_info.class_bbox_count += int(count)
        self.total_bbox_count += int(count)
        if self.total_bbox_count > 0:
            for ci in self.class_distribution_infos:
                ci.class_bbox_percentage = 100 * (ci.class_bbox_count / self.total_bbox_count)

    def join(self, other: "ClassDistribution") -> "ClassDistribution":
        class_dist = ClassDistribution()
        for class_info in self.class_distribution_infos + other.class_distribution_infos:
            class_dist.add_pixel_count(class_name=class_info.class_name, count=class_info.class_pixel_count)
            class_dist.add_instance_count(class_name=class_info.class_name, count=class_info.class_instance_count)
            class_dist.add_bbox_count(class_name=class_info.class_name, count=class_info.class_bbox_count)
        return class_dist

    def update(
        self,
        other: "ClassDistribution",
        annotation_types_to_use: Optional[Union[AnnotationTypes, List[AnnotationTypes]]] = None,
    ):
        if annotation_types_to_use is None:
            annotation_types_to_use = [AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D]
        elif isinstance(annotation_types_to_use, AnnotationTypes):
            annotation_types_to_use = [annotation_types_to_use]
        for class_info in other.class_distribution_infos:
            if AnnotationTypes.SemanticSegmentation2D in annotation_types_to_use:
                self.add_pixel_count(class_name=class_info.class_name, count=class_info.class_pixel_count)
            if AnnotationTypes.InstanceSegmentation2D in annotation_types_to_use:
                self.add_instance_count(class_name=class_info.class_name, count=class_info.class_instance_count)
            if AnnotationTypes.BoundingBoxes2D in annotation_types_to_use:
                self.add_bbox_count(class_name=class_info.class_name, count=class_info.class_bbox_count)

    @staticmethod
    def from_dataset(
        dataset: Dataset,
        sensors_to_use: Optional[Set[SensorName]] = None,
        annotation_types_to_use: Optional[Union[AnnotationTypes, List[AnnotationTypes]]] = None,
    ) -> "ClassDistribution":
        if annotation_types_to_use is None:
            annotation_types_to_use = [AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D]
        elif isinstance(annotation_types_to_use, AnnotationTypes):
            annotation_types_to_use = [annotation_types_to_use]

        class_dist = ClassDistribution()
        for scene_name, scene in dataset.unordered_scenes.items():
            class_dist.update(
                ClassDistribution.from_scene(
                    scene=scene, sensors_to_use=sensors_to_use, annotation_types_to_use=annotation_types_to_use
                ),
                annotation_types_to_use=annotation_types_to_use,
            )
            print(f"{scene_name} complete.")
        return class_dist

    @staticmethod
    def from_scene(
        scene: UnorderedScene,
        sensors_to_use: Optional[Set[SensorName]] = None,
        annotation_types_to_use: Optional[Union[AnnotationTypes, List[AnnotationTypes]]] = None,
    ) -> "ClassDistribution":
        if annotation_types_to_use is None:
            annotation_types_to_use = [AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D]
        elif isinstance(annotation_types_to_use, AnnotationTypes):
            annotation_types_to_use = [annotation_types_to_use]

        if sensors_to_use is not None:
            cam_names = sensors_to_use.intersection(scene.camera_names)
        else:
            cam_names = scene.camera_names

        return ClassDistribution.from_cameras(
            cameras=(scene.get_sensor(sensor_name=s) for s in cam_names),
            class_maps=scene.class_maps,
            annotation_types_to_use=annotation_types_to_use,
        )

    @staticmethod
    def from_cameras(
        cameras: Generator[CameraSensor, None, None],
        class_maps: Dict[AnnotationType, ClassMap],
        annotation_types_to_use: Optional[Union[AnnotationTypes, List[AnnotationTypes]]] = None,
    ) -> "ClassDistribution":
        class_dist = ClassDistribution()

        if AnnotationTypes.SemanticSegmentation2D in annotation_types_to_use:
            semseg_class_map = class_maps[AnnotationTypes.SemanticSegmentation2D]
        if AnnotationTypes.BoundingBoxes2D in annotation_types_to_use:
            bbox_class_map = class_maps[AnnotationTypes.BoundingBoxes2D]
        for camera in cameras:
            for frame_id in camera.frame_ids:
                camera_frame = camera.get_frame(frame_id=frame_id)
                if (AnnotationTypes.SemanticSegmentation2D in camera_frame.available_annotation_types) and (
                    AnnotationTypes.SemanticSegmentation2D in annotation_types_to_use
                ):
                    semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
                    class_dist.update_with_semantic_segmentation_2d(
                        semantic_segmentation=semseg, class_map=semseg_class_map
                    )
                    if (AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types) and (
                        AnnotationTypes.InstanceSegmentation2D in annotation_types_to_use
                    ):
                        instseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
                        class_dist.update_with_instance_segmentation_2d(
                            instance_segmentation=instseg, class_map=semseg_class_map, semantic_segmentation=semseg
                        )
                if (AnnotationTypes.BoundingBoxes2D in camera_frame.available_annotation_types) and (
                    AnnotationTypes.BoundingBoxes2D in annotation_types_to_use
                ):
                    bboxes = camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
                    class_dist.update_with_bounding_box(bboxes=bboxes, class_map=bbox_class_map)
        return class_dist

    def update_with_semantic_segmentation_2d(self, semantic_segmentation: SemanticSegmentation2D, class_map: ClassMap):
        class_ids = semantic_segmentation.class_ids
        values, counts = np.unique(class_ids, return_counts=True)
        for class_id, class_cnt in zip(values, counts):
            class_name = class_map[class_id].name
            self.add_pixel_count(class_name=class_name, count=class_cnt)

    def update_with_instance_segmentation_2d(
        self,
        instance_segmentation: InstanceSegmentation2D,
        semantic_segmentation: SemanticSegmentation2D,
        class_map: ClassMap,
    ):
        instance_ids = instance_segmentation.instance_ids
        values, counts = np.unique(instance_ids, return_counts=True)
        for instance_id, instance_cnt in zip(values, counts):
            mask = instance_segmentation.get_instance(instance_id=instance_id)
            class_ids = semantic_segmentation.class_ids[mask]
            class_ids, class_counts = np.unique(class_ids, return_counts=True)
            max_cnt = np.max(class_counts)
            class_id = class_ids[list(class_counts).index(max_cnt)]
            class_info = class_map[class_id]
            if class_info.instanced:
                class_name = class_info.name
                self.add_instance_count(class_name=class_name, count=1)

    def update_with_bounding_box(
        self,
        bboxes: BoundingBoxes2D,
        class_map: ClassMap,
    ):
        bbox_ids = [box.class_id for box in bboxes.boxes]
        bbox_class_counts = Counter(bbox_ids)
        for class_id, class_count in bbox_class_counts.items():
            class_name = class_map[class_id].name
            self.add_bbox_count(class_name=class_name, count=class_count)
