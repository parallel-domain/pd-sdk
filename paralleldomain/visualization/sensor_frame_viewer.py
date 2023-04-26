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
        img = sensor_frame.image.rgb
        img = img[..., ::-1]

        # show image
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Image", img)
        cv2.waitKey(wait_time)

        # for any annotation types specified, show the annotation type
        for annotype in annotations_to_show:
            if annotype is AnnotationTypes.BoundingBoxes2D:
                img = np.ascontiguousarray(img, dtype=np.uint8)
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

                # show modified image
                cv2.namedWindow("Bounding Boxes 2D", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Bounding Boxes 2D", img)
                cv2.waitKey(wait_time)

            elif annotype is AnnotationTypes.SemanticSegmentation2D:
                # get semseg mask
                class_ids = sensor_frame.get_annotations(annotation_type=annotype).class_ids

                # get number of classes
                classes = np.unique(class_ids)

                # define colormap
                np.random.seed(42)
                colors = np.random.randint(0, 255, (len(classes), 3))
                colormap = dict(zip(classes, colors))

                # create an empty output image
                h, w = class_ids.shape[:2]
                color_image = np.zeros((h, w, 3), dtype=np.uint8)

                # fill in image with colormapped values
                for i in range(h):
                    for j in range(w):
                        # mod just in case
                        color_image[i, j] = colormap[int(class_ids[i, j])]

                # show semseg mask
                cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Semantic Segmentation", color_image)
                cv2.waitKey(wait_time)

            elif annotype is AnnotationTypes.InstanceSegmentation2D:
                # get instance seg seg mask
                instance_ids = sensor_frame.get_annotations(annotation_type=annotype).instance_ids

                # determine the number of unique instance IDs
                num_instances = len(np.unique(instance_ids))

                # create a random colormap with enough entries to cover the full range of instance IDs
                np.random.seed(42)
                colormap_size = max(num_instances, 256)
                colormap = np.random.randint(0, 256, size=(colormap_size, 3), dtype=np.uint8)

                # map the instance IDs to their corresponding colors
                instance_colors = colormap[instance_ids.squeeze() % colormap_size]

                # replace the RGB values with the instance colors wherever the instance IDs are non-zero
                instance_mask = instance_ids.squeeze() > 0
                img[instance_mask] = instance_colors[instance_mask]

                # show modified image
                cv2.namedWindow("Instance Segmentation", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Instance Segmentation", img)
                cv2.waitKey(wait_time)

            elif annotype is AnnotationTypes.Depth:
                # get depth mask
                depth_image = sensor_frame.get_annotations(annotation_type=annotype).depth

                # use a continuous colormap
                colormap = cv2.COLORMAP_JET
                depth_range = 100.0
                depth_max_range = 1000.0

                # clip depth values past max range
                depth_image[depth_image > depth_max_range] = depth_max_range

                # normalize depth values to 0-1 range
                depth_float = depth_image.astype(np.float32) / depth_range

                # wrap depth values
                depth_wrapped = depth_float % 1.0

                colormap_image = cv2.applyColorMap((depth_wrapped * 255).astype(np.uint8), colormap)

                depth_image = cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB)

                # show depth
                cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Depth", depth_image)
                cv2.waitKey(wait_time)
    else:
        pass
        # raise ValueError()
