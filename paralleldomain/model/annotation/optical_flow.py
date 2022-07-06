from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class OpticalFlow(Annotation):
    """Represents an Optical Flow mask for a camera image.

    Args:
        vectors: :attr:`paralleldomain.model.annotation.optical_flow.OpticalFlow.vectors`

    Attributes:
        vectors: Matrix of shape `(H X W x 2)`, where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the x and y offset to the pixels coordinate on the next image.

    Example:
        Using the Optical Flow vector mask in combination with :attr:`.Image.coordinates` allows for a
        fast retrieval of absolute pixel coordinates.
        ::

            camera_frame: SensorFrame = ...  # get any camera's SensorFrame

            flow = camera_frame.get_annotations(AnnotationTypes.OpticalFlow)
            rgb = camera_frame.image.rgb
            next_image = np.zeros_like(rgb)
            coordinates = camera_frame.image.coordinates
            next_frame_coords = coordinates + np.round(flow.vectors).astype(int)[...,[1,0]]

            for y in range(rgb.shape[0]):
                for x in range(rgb.shape[1]):
                    next_coord = next_frame_coords[y, x]
                    if 0 <= next_coord[0] < rgb.shape[0] and 0 <= next_coord[1] < rgb.shape[1]:
                        next_image[next_coord[0], next_coord[1], :] = rgb[y, x, :]

            import cv2
            cv2.imshow("window_name", cv2.cvtColor(
                    src=next_image,
                    code=cv2.COLOR_RGBA2BGRA,
            ))
            cv2.waitKey()
    """

    vectors: np.ndarray

    def __sizeof__(self):
        return self.vectors.nbytes
