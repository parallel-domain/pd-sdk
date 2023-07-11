import cv2
import pickle
import numpy as np

from typing import List, Dict, Optional
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.annotation import BoundingBoxes2D, SemanticSegmentation2D
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY

MAX_IMAGE_EXTENT = 450


def maintain_aspect_ratio(width, height, max_edge_length):
    aspect_ratio = width / height

    if width <= max_edge_length and height <= max_edge_length:
        return width, height

    if width > max_edge_length:
        new_width = max_edge_length
        new_height = new_width / aspect_ratio
    else:
        new_height = max_edge_length
        new_width = new_height * aspect_ratio

    return int(new_width), int(new_height)


@STATISTICS_REGISTRY.register_module()
class ClassHeatMaps(Statistic):
    def __init__(self, classes_of_interest: List[str] = None) -> None:
        super().__init__()
        self._heat_maps: Dict[str, np.ndarray] = {}
        self._classes_of_interest = classes_of_interest

    def _reset(self):
        self._heat_maps.clear()

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        height = sensor_frame.image.height
        width = sensor_frame.image.width

        width, height = maintain_aspect_ratio(width=width, height=height, max_edge_length=MAX_IMAGE_EXTENT)
        resize_factor = width / sensor_frame.image.width

        if SemanticSegmentation2D in sensor_frame.available_annotation_types:
            semseg_2d: SemanticSegmentation2D = sensor_frame.get_annotations(annotation_type=SemanticSegmentation2D)
            class_map = sensor_frame.class_maps[SemanticSegmentation2D]

            for class_name in class_map.class_names:
                if self._classes_of_interest is not None and class_name not in self._classes_of_interest:
                    continue
                self._heat_maps.setdefault(class_name, np.zeros(shape=(height, width), dtype=np.uint32))

            class_ids = np.squeeze(semseg_2d.class_ids)
            class_ids = cv2.resize(src=class_ids, dsize=[width, height], interpolation=cv2.INTER_NEAREST)

            for class_id in np.unique(class_ids):
                class_name = class_map[class_id].name
                if self._classes_of_interest is not None and class_name not in self._classes_of_interest:
                    continue
                class_heat_map = self._heat_maps.get(class_name, np.zeros(shape=(height, width), dtype=np.uint32))
                class_heat_map[class_ids == class_id] += 1
                self._heat_maps[class_name] = class_heat_map
        elif BoundingBoxes2D in sensor_frame.available_annotation_types:
            bboxes = sensor_frame.get_annotations(annotation_type=BoundingBoxes2D)
            class_map = sensor_frame.class_maps[BoundingBoxes2D]

            for class_name in class_map.class_names:
                if self._classes_of_interest is not None and class_name not in self._classes_of_interest:
                    continue
                self._heat_maps.setdefault(class_name, np.zeros(shape=(height, width), dtype=np.uint32))

            for bbox in bboxes.boxes:
                class_name = class_map[bbox.class_id].name
                if self._classes_of_interest is not None and class_name not in self._classes_of_interest:
                    continue
                class_heat_map = self._heat_maps.get(class_name, np.zeros(shape=(height, width), dtype=np.uint32))
                bbox_y_min = int(bbox.y_min * resize_factor)
                bbox_y_max = int(bbox.y_max * resize_factor)
                bbox_x_min = int(bbox.x_min * resize_factor)
                bbox_x_max = int(bbox.x_max * resize_factor)
                class_heat_map[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max] += 1
                self._heat_maps[class_name] = class_heat_map
        else:
            raise Exception(f"No suitable annotation found to compute {self.__class__.__name__}")

    def _load(self, file_path: str):
        with open(file_path, "rb") as f:
            self._heat_maps = pickle.load(f)

    def _save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self._heat_maps, f)

    def get_classes(self) -> List[str]:
        return list(self._heat_maps.keys())

    def get_heatmaps(self, classes_of_interest: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        return {
            class_name: heat_map
            for class_name, heat_map in self._heat_maps.items()
            if (classes_of_interest is None or class_name in classes_of_interest)
        }
