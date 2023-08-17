from collections import defaultdict
import pickle

import logging
from typing import Union

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.annotation import BoundingBoxes2D
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY
from paralleldomain.utilities.any_path import AnyPath


@STATISTICS_REGISTRY.register_module()
class Aggregated2DBoundingBoxAnnotations(Statistic):
    """
    Class that handles parsing and loading/saving pixels counts from 2D bounding box annotations.
    """

    def __init__(self) -> None:
        super().__init__()

        self.reset()

    def _reset(self):
        self._recorder = defaultdict(list)

    def _record_empty_frames(self, scene: Scene, sensor_frame: CameraSensorFrame):
        self._recorder["skipped_frames"].append(self.parse_sensor_frame_properties(scene, sensor_frame))

    @staticmethod
    def parse_2d_bounding_box_annotations(sensor_frame: CameraSensorFrame):
        bbox_2d_annotations = {}

        bbox_2d_annotations["img_height"] = sensor_frame.image.height
        bbox_2d_annotations["img_width"] = sensor_frame.image.width

        bbox_2d: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
        class_map = sensor_frame.class_maps[BoundingBoxes2D]

        boxes = []
        for box in bbox_2d.boxes:
            class_detail = class_map[box.class_id]
            box_dict = {
                "class_name": class_detail.name,
                "class_id": box.class_id,
                "instance_id": box.instance_id,
                "x": box.x,
                "y": box.y,
                "height": box.height,
                "width": box.width,
                "attributes": box.attributes,
            }
            boxes.append(box_dict)
        bbox_2d_annotations["bboxes_2d"] = boxes

        return bbox_2d_annotations

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        if BoundingBoxes2D in sensor_frame.available_annotation_types:
            bbox_2d_annotations = self.parse_2d_bounding_box_annotations(sensor_frame=sensor_frame)
            # add the properties_to_log to the annotations we want to save.
            bbox_2d_annotations.update(self.parse_sensor_frame_properties(scene, sensor_frame))
            self._recorder["bbox_2d_annotations"].append(bbox_2d_annotations)
        else:
            logging.warning("No 2D bounding box annotations available for current frame... Logging and skipping")
            self._record_empty_frames(scene=scene, sensor_frame=sensor_frame)

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("rb") as f:
            self._recorder = pickle.load(f)

    def _save(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("wb") as f:
            pickle.dump(self._recorder, f, protocol=pickle.HIGHEST_PROTOCOL)
