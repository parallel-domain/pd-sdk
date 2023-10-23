from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class OpticalFlow(Annotation):
    """
    Represents an Optical Flow mask for a camera image.

    Args:
        vectors: :attr:`OpticalFlow.vectors`
        valid_mask: :attr:`OpticalFlow.valid_mask`

    Attributes:
        vectors: Matrix of shape `(H X W x 2)`, where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the x and y offset to the pixels coordinate on the next image.
        valid_mask: Matrix of shape `(H X W)` with values {0,1}. 1 indicates a pixel with a valid flow label in the
            vectors attribute. 0 indicates no ground truth flow at that pixel.

    Example:
        Using the Optical Flow vector mask in combination with :attr:`.Image.coordinates` allows for a
        fast retrieval of absolute pixel coordinates::

            >>> camera_frame: SensorFrame = ...  # get any camera's SensorFrame
            >>>
            >>> flow = camera_frame.get_annotations(AnnotationTypes.OpticalFlow)
            >>> rgb = camera_frame.image.rgb
            >>> next_image = np.zeros_like(rgb)
            >>> coordinates = camera_frame.image.coordinates
            >>> next_frame_coords = coordinates + np.round(flow.vectors).astype(int)[...,[1,0]]
            >>>
            >>> for y in range(rgb.shape[0]):
            >>>     for x in range(rgb.shape[1]):
            >>>         next_coord = next_frame_coords[y, x]
            >>>         if 0 <= next_coord[0] < rgb.shape[0] and 0 <= next_coord[1] < rgb.shape[1]:
            >>>             next_image[next_coord[0], next_coord[1], :] = rgb[y, x, :]
            >>>
            >>> import cv2
            >>> cv2.imshow("window_name", cv2.cvtColor(
            >>>         src=next_image,
            >>>         code=cv2.COLOR_RGBA2BGRA,
            >>> ))
            >>> cv2.waitKey()
    """

    vectors: np.ndarray
    valid_mask: Optional[np.ndarray] = None

    def __sizeof__(self):
        total_size = 0
        for vec in [self.vectors, self.valid_mask]:
            if vec is not None:
                total_size += vec.nbytes
        return total_size
