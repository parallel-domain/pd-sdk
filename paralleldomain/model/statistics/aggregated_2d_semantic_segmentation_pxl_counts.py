from collections import defaultdict
import pickle

import logging
from typing import Union

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.annotation import SemanticSegmentation2D
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.model.image import Image


@STATISTICS_REGISTRY.register_module()
class Aggregated2DSemanticSegmentationPixelCounts(Statistic):
    """
    Class that handles parsing and loading/saving 2D bounding box annotations and metadata.
    """

    def __init__(self) -> None:
        super().__init__()

        self.reset()

    def _reset(self):
        self._recorder = defaultdict(list)

    def _record_empty_frames(self, scene: Scene, sensor_frame: CameraSensorFrame):
        self._recorder["skipped_frames"].append(self.parse_sensor_frame_properties(scene, sensor_frame))

    @staticmethod
    def collect_2d_semantic_segmentation_pixel_counts(sensor_frame: CameraSensorFrame):
        semseg_2d_pixel_counts = {
            "img_height": sensor_frame.image.height,
            "img_width": sensor_frame.image.width,
            "img_filepath": sensor_frame.get_file_path(Image),
        }

        semseg_2d: SemanticSegmentation2D = sensor_frame.get_annotations(annotation_type=SemanticSegmentation2D)
        class_map = sensor_frame.class_maps[SemanticSegmentation2D]
        pixel_counts_per_class = {}

        for c_id, c_name in zip(class_map.class_ids, class_map.class_names):
            pixel_count = semseg_2d.get_class_mask(c_id).sum()
            if pixel_count:
                pixel_counts_per_class[c_name] = pixel_count

        semseg_2d_pixel_counts["semseg_2d_pixel_counts"] = pixel_counts_per_class

        return semseg_2d_pixel_counts

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        if SemanticSegmentation2D in sensor_frame.available_annotation_types:
            semantic_segmentation_pixel_counts = self.collect_2d_semantic_segmentation_pixel_counts(sensor_frame)
            # add the properties_to_log to the annotations we want to save.
            semantic_segmentation_pixel_counts.update(self.parse_sensor_frame_properties(scene, sensor_frame))
            self._recorder["semseg_2d_annotations"].append(semantic_segmentation_pixel_counts)
        else:
            logging.warning("No 2D semantic segmentation annotations available for current frame... Skipping")
            self._record_empty_frames(scene=scene, sensor_frame=sensor_frame)

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("rb") as f:
            self._recorder = pickle.load(f)

    def _save(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("wb") as f:
            pickle.dump(self._recorder, f, protocol=pickle.HIGHEST_PROTOCOL)
