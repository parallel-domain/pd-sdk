from typing import List

import cv2
import numpy as np

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, BoundingBoxes2D
from paralleldomain.model.sensor import CameraSensorFrame

DEFAULT_COLOR = (0, 255, 0)


def show_sensor_frame(
    sensor_frame: CameraSensorFrame, annotations_to_show: List[AnnotationType] = None, frames_per_second: float = 2
):
    annotations_to_show = list() if annotations_to_show is None else annotations_to_show
    wait_time = int(1000 / frames_per_second)
    if isinstance(sensor_frame, CameraSensorFrame):
        img = sensor_frame.image.rgb[..., ::-1]
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for annotype in annotations_to_show:
            if annotype is AnnotationTypes.BoundingBoxes2D:
                boxes: BoundingBoxes2D = sensor_frame.get_annotations(annotation_type=annotype)

                class_map = sensor_frame.class_maps[annotype]

                for box in boxes:
                    class_detail = class_map[box.class_id]
                    color = DEFAULT_COLOR
                    if "color" in class_detail.meta:
                        color = class_detail.meta["color"]
                        color = (color["r"], color["g"], color["b"])
                    cv2.rectangle(img, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2)
                    cv2.putText(img, class_detail.name, (box.x_min, box.y_min - 4), 0, 0.3, color)

        cv2.imshow("Sensor Frame", img)
        cv2.waitKey(wait_time)
    else:
        pass
        # raise ValueError()
