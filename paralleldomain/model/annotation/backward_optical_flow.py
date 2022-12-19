from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class BackwardOpticalFlow(Annotation):
    """Represents a Backwards Optical Flow mask for a camera image.
    Args:
        vectors: :attr:`paralleldomain.model.annotation.backwards_optical_flow.BackwardsOpticalFlow.vectors`
    Attributes:
        vectors: Matrix of shape `(H X W x 2)`, where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the x and y offset to the pixels coordinate on the previous image.
    Example:
        Using the Optical Flow vector mask in combination with :attr:`.Image.coordinates` allows for a
        fast retrieval of absolute pixel coordinates.
        ::
            camera_frame: SensorFrame = ...  # get any camera's SensorFrame
            flow = camera_frame.get_annotations(AnnotationTypes.BackwardOpticalFlow)
            rgb = camera_frame.image.rgb
            prev_image = np.zeros_like(rgb)
            coordinates = camera_frame.image.coordinates
            prev_frame_coords = coordinates + np.round(flow.vectors).astype(int)[...,[1,0]]
            for y in range(rgb.shape[0]):
                for x in range(rgb.shape[1]):
                    prev_coord = prev_frame_coords[y, x]
                    if 0 <= prev_coord[0] < rgb.shape[0] and 0 <= prev_coord[1] < rgb.shape[1]:
                        prev_image[prev_coord[0], prev_coord[1], :] = rgb[y, x, :]
            import cv2
            cv2.imshow("window_name", cv2.cvtColor(
                    src=prev_image,
                    code=cv2.COLOR_RGBA2BGRA,
            ))
            cv2.waitKey()
    """

    vectors: np.ndarray
    valid_mask: Optional[np.ndarray] = None

    def __sizeof__(self):
        total_size = 0
        for vec in [self.vectors, self.valid_mask]:
            if vec is not None:
                total_size += vec.nbytes
        return total_size
