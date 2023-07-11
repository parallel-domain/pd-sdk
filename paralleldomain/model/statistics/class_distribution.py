import numpy as np

from typing import Optional

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.annotation import (
    AnnotationTypes,
    BoundingBoxes2D,
    BoundingBoxes3D,
    SemanticSegmentation2D,
    InstanceSegmentation2D,
)
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY


try:
    import pandas
except ImportError:
    _has_pandas = False
else:
    _has_pandas = True


@STATISTICS_REGISTRY.register_module()
class ClassDistribution(Statistic):
    """
    Implements a class counter that stores per sensor frame information about class distributions.
    """

    def __init__(self) -> None:
        if not _has_pandas:
            raise ImportError(
                f"pandas is required to compute use {ClassDistribution.__name__}."
                "Please install optional 'statistics' dependency e.g. 'pip install .[statistics]'"
            )

        super().__init__()

        self.reset()

    def _reset(self):
        self._class_counter_df = pandas.DataFrame(
            columns=["scene_name", "frame_id", "sensor_name", "class_name", "num_instances", "num_pixels"]
        )

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        pandas_row_per_class = {}

        if SemanticSegmentation2D in sensor_frame.available_annotation_types:
            class_map = sensor_frame.class_maps[SemanticSegmentation2D]

            semseg_2d: SemanticSegmentation2D = sensor_frame.get_annotations(annotation_type=SemanticSegmentation2D)

            for class_id in np.unique(semseg_2d.class_ids):
                class_detail = class_map[class_id]

                class_mask = semseg_2d.class_ids == class_id
                row = pandas_row_per_class.setdefault(
                    class_detail.name,
                    {
                        "scene_name": scene.name,
                        "frame_id": sensor_frame.frame_id,
                        "sensor_name": sensor_frame.sensor_name,
                        "num_instances": 0,
                        "num_pixels": 0,
                    },
                )
                row["num_pixels"] += np.sum(class_mask)

                if class_detail.instanced and InstanceSegmentation2D in sensor_frame.available_annotation_types:
                    instance_2d: InstanceSegmentation2D = sensor_frame.get_annotations(
                        annotation_type=InstanceSegmentation2D
                    )
                    row["num_instances"] += len(np.unique(instance_2d.instance_ids[class_mask]))

                elif BoundingBoxes2D in sensor_frame.available_annotation_types:
                    bbox_2d: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
                    row["num_instances"] += len([box for box in bbox_2d.boxes if box.class_id == class_id])

        elif BoundingBoxes2D in sensor_frame.available_annotation_types:
            bbox_2d: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
            class_map = sensor_frame.class_maps[BoundingBoxes2D]

            for box in bbox_2d.boxes:
                class_detail = class_map[box.class_id]
                row = pandas_row_per_class.setdefault(
                    class_detail.name,
                    {
                        "scene_name": scene.name,
                        "frame_id": sensor_frame.frame_id,
                        "sensor_name": sensor_frame.sensor_name,
                        "num_instances": 0,
                        "num_pixels": 0,
                    },
                )
                row["num_instances"] += 1
                row["num_pixels"] += box.width * box.height

        elif AnnotationTypes.BoundingBoxes3D in sensor_frame.available_annotation_types:
            bbox_3d: BoundingBoxes3D = sensor_frame.get_annotations(annotation_type=BoundingBoxes3D)
            class_map = sensor_frame.class_maps[BoundingBoxes3D]

            for box in bbox_3d.boxes:
                class_detail = class_map[box.class_id]
                row = pandas_row_per_class.setdefault(
                    class_detail.name,
                    {
                        "scene_name": scene.name,
                        "frame_id": sensor_frame.frame_id,
                        "sensor_name": sensor_frame.sensor_name,
                        "num_instances": 0,
                        "num_pixels": 0,
                    },
                )
                row["num_instances"] += 1
        else:
            raise Exception("No suitable annotation available to compute class distribution.")

        self._class_counter_df = self._class_counter_df.append(
            list(dict(class_name=key, **value) for key, value in pandas_row_per_class.items()), ignore_index=True
        )

    def _load(self, file_path: str):
        self._class_counter_df = pandas.read_pickle(file_path)

    def _save(self, file_path: str):
        self._class_counter_df.to_pickle(file_path)

    def number_of_instances(self, class_name: str, scene_name: Optional[str] = None, sensor_name: Optional[str] = None):
        df = self._class_counter_df
        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.query("num_instances > 0")
        counts = counts.groupby("class_name")["num_instances"].sum()
        return counts[class_name]

    def number_of_pixels(self, class_name: str, scene_name: Optional[str] = None, sensor_name: Optional[str] = None):
        df = self._class_counter_df
        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.groupby("class_name")["num_pixels"].sum()
        return counts[class_name]

    def get_instance_distribution(self, scene_name: Optional[str] = None, sensor_name: Optional[str] = None):
        df = self._class_counter_df
        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.query("num_instances > 0")
        counts = counts.groupby("class_name")["num_instances"].sum()
        return counts.to_dict()

    def get_pixel_distribution(self, scene_name: Optional[str] = None, sensor_name: Optional[str] = None):
        df = self._class_counter_df
        if scene_name is not None:
            df = df.query(f"scene_name == '{scene_name}'")
        if sensor_name is not None:
            df = df.query(f"sensor_name == '{sensor_name}'")
        counts = df.groupby("class_name")["num_pixels"].sum()
        return counts.to_dict()
