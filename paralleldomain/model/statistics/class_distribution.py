from collections import defaultdict
import numpy as np
import logging
import pickle

from typing import Dict, Optional, List, Union

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.annotation import (
    BoundingBoxes2D,
    BoundingBoxes3D,
    SemanticSegmentation2D,
    InstanceSegmentation2D,
)
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.model.annotation.common import AnnotationType


try:
    import pandas
except ImportError:
    _has_pandas = False
else:
    _has_pandas = True


@STATISTICS_REGISTRY.register_module(name="class_distribution")
class ClassDistribution(Statistic):
    """
    Implements a class counter that stores per sensor frame information about class distributions.
    """

    def __init__(self, source_annotation_types: List[AnnotationType] = [SemanticSegmentation2D]) -> None:
        if not _has_pandas:
            raise ImportError(
                f"pandas is required to compute use {ClassDistribution.__name__}."
                "Please install optional 'statistics' dependency e.g. 'pip install .[statistics]'"
            )

        self.SUPPORTED_ANNOTATION_TYPES = [SemanticSegmentation2D, BoundingBoxes2D, BoundingBoxes3D]

        if not isinstance(source_annotation_types, list):
            source_annotation_types = [source_annotation_types]

        for annotation_type in source_annotation_types:
            assert (
                annotation_type in self.SUPPORTED_ANNOTATION_TYPES
            ), f"{annotation_type} not in supported annotation types: {self.SUPPORTED_ANNOTATION_TYPES}"

        self.source_annotation_types = source_annotation_types

        super().__init__()
        self.reset()

    def _reset(self):
        self._recorder = defaultdict(list)
        # We are going to create the dataframe with the annotation type keys to avoid key errors
        # when wanting to query the created dataframe later during visualization.
        for x in self.source_annotation_types:
            self._recorder[x.__name__] = []

    def _record_empty_frames(self, scene: Scene, sensor_frame: CameraSensorFrame):
        self._recorder["skipped_frames"].append(self.parse_sensor_frame_properties(scene, sensor_frame))

    def _add_to_counter(self, pixel_counts_per_class: Dict, current_annotation_type: str):
        # reformat to list of dicts
        pixel_counts_per_class = [dict(class_name=key, **value) for key, value in pixel_counts_per_class.items()]
        for d in pixel_counts_per_class:
            self._recorder[current_annotation_type].append(d)

    def collect_pixel_counts_from_semantic_segmentation_2d(self, scene: Scene, sensor_frame: CameraSensorFrame):
        current_annotation_type = SemanticSegmentation2D.__name__

        pixel_counts_per_class = {}
        class_map = sensor_frame.class_maps[SemanticSegmentation2D]
        semseg_2d: SemanticSegmentation2D = sensor_frame.get_annotations(annotation_type=SemanticSegmentation2D)

        for class_id in np.unique(semseg_2d.class_ids):
            class_detail = class_map[class_id]

            class_mask = semseg_2d.class_ids == class_id
            class_pxl_cnts_on_frame = pixel_counts_per_class.setdefault(
                class_detail.name, {"num_instances": 0, "num_pixels": 0}
            )
            class_pxl_cnts_on_frame["num_pixels"] += np.sum(class_mask)

            if class_detail.instanced and InstanceSegmentation2D in sensor_frame.available_annotation_types:
                instance_2d: InstanceSegmentation2D = sensor_frame.get_annotations(
                    annotation_type=InstanceSegmentation2D
                )
                class_pxl_cnts_on_frame["num_instances"] += len(np.unique(instance_2d.instance_ids[class_mask]))

            elif BoundingBoxes2D in sensor_frame.available_annotation_types:
                bbox_2d: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
                class_pxl_cnts_on_frame["num_instances"] += len(
                    [box for box in bbox_2d.boxes if box.class_id == class_id]
                )

            class_pxl_cnts_on_frame["from_source"] = "SemanticSegmentation2D"
            class_pxl_cnts_on_frame.update(self.parse_sensor_frame_properties(scene, sensor_frame))

        self._add_to_counter(
            pixel_counts_per_class=pixel_counts_per_class, current_annotation_type=current_annotation_type
        )

    def collect_pixel_counts_from_bounding_boxes_2d(self, scene: Scene, sensor_frame: CameraSensorFrame):
        current_annotation_type = BoundingBoxes2D.__name__
        pixel_counts_per_class = {}
        bboxes_2d: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
        class_map = sensor_frame.class_maps[BoundingBoxes2D]

        for box in bboxes_2d.boxes:
            class_detail = class_map[box.class_id]
            class_pxl_cnts_on_frame = pixel_counts_per_class.setdefault(
                class_detail.name, {"num_instances": 0, "num_pixels": 0}
            )
            class_pxl_cnts_on_frame["num_instances"] += 1
            class_pxl_cnts_on_frame["num_pixels"] += box.width * box.height
            class_pxl_cnts_on_frame["box_height"] = box.height
            class_pxl_cnts_on_frame["box_width"] = box.width
            class_pxl_cnts_on_frame["from_source"] = "BoundingBoxes2D"
            class_pxl_cnts_on_frame.update(self.parse_sensor_frame_properties(scene, sensor_frame))

        self._add_to_counter(
            pixel_counts_per_class=pixel_counts_per_class, current_annotation_type=current_annotation_type
        )

    def collect_pixel_counts_from_bounding_boxes_3d(self, scene: Scene, sensor_frame: CameraSensorFrame):
        current_annotation_type = BoundingBoxes3D.__name__
        pixel_counts_per_class = {}
        bbox_3d: BoundingBoxes3D = sensor_frame.get_annotations(annotation_type=BoundingBoxes3D)
        class_map = sensor_frame.class_maps[BoundingBoxes3D]

        for box in bbox_3d.boxes:
            class_detail = class_map[box.class_id]
            class_pxl_cnts_on_frame = pixel_counts_per_class.setdefault(
                class_detail.name, {"num_instances": 0, "num_pixels": 0}
            )
            class_pxl_cnts_on_frame["num_instances"] += 1
            class_pxl_cnts_on_frame["from_source"] = "BoundingBoxes3D"
            class_pxl_cnts_on_frame.update(self.parse_sensor_frame_properties(scene, sensor_frame))

        self._add_to_counter(
            pixel_counts_per_class=pixel_counts_per_class, current_annotation_type=current_annotation_type
        )

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("rb") as f:
            self._recorder = pickle.load(f)

    def _save(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("wb") as f:
            pickle.dump(self._recorder, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        if len(sensor_frame.available_annotation_types) == 0:
            self._record_empty_frames(scene=scene, sensor_frame=sensor_frame)
            logging.warning("No avaliable annotation types in current frame... Skipping")
            return

        for current_annotation_type in self.source_annotation_types:
            if current_annotation_type not in sensor_frame.available_annotation_types:
                logging.warning(f"{current_annotation_type} not available for current frame... Skipping")
                return

            # Could potentially clean this up with some type of registry call.
            if current_annotation_type == SemanticSegmentation2D:
                self.collect_pixel_counts_from_semantic_segmentation_2d(scene, sensor_frame)

            if current_annotation_type == BoundingBoxes2D:
                self.collect_pixel_counts_from_bounding_boxes_2d(scene, sensor_frame)

            if current_annotation_type == BoundingBoxes3D:
                self.collect_pixel_counts_from_bounding_boxes_3d(scene, sensor_frame)

    def number_of_instances(
        self,
        class_name: str,
        annotation_type: AnnotationType = SemanticSegmentation2D,
        scene_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
    ):
        annotation_type = annotation_type.__name__

        assert (
            annotation_type in self.source_annotation_types
        ), f"{annotation_type} is not part of source_annotation_types"

        if len(self._recorder[annotation_type]) == 0:
            return pandas.DataFrame()

        df = pandas.DataFrame(self._recorder[annotation_type])

        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.query("num_instances > 0")
        counts = counts.groupby("class_name")["num_instances"].sum()
        return counts[class_name]

    def number_of_pixels(
        self,
        class_name: str,
        annotation_type: AnnotationType = SemanticSegmentation2D,
        scene_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
    ):
        annotation_type = annotation_type.__name__
        assert (
            annotation_type in self.source_annotation_types
        ), f"{annotation_type} is not part of source_annotation_types"

        if len(self._recorder[annotation_type]) == 0:
            return {}

        df = pandas.DataFrame(self._recorder[annotation_type])

        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.groupby("class_name")["num_pixels"].sum()
        return counts[class_name]

    def get_instance_distribution(
        self,
        annotation_type: AnnotationType = SemanticSegmentation2D,
        scene_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
    ):
        assert (
            annotation_type in self.source_annotation_types
        ), f"{annotation_type.__name__} is not part of source_annotation_types"
        annotation_type = annotation_type.__name__

        if len(self._recorder[annotation_type]) == 0:
            return {}

        df = pandas.DataFrame(self._recorder[annotation_type])

        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.query("num_instances > 0")
        counts = counts.groupby("class_name")["num_instances"].sum()
        return counts.to_dict()

    def get_pixel_distribution(
        self,
        annotation_type: AnnotationType = SemanticSegmentation2D,
        scene_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
    ):
        assert (
            annotation_type in self.source_annotation_types
        ), f"{annotation_type.__name__} is not part of source_annotation_types"
        annotation_type = annotation_type.__name__

        if len(self._recorder[annotation_type]) == 0:
            return {}

        df = pandas.DataFrame(self._recorder[annotation_type])

        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.groupby("class_name")["num_pixels"].sum()
        return counts.to_dict()
