from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class Depth(Annotation):
    """Represents a Depth mask for a camera image.

    Args:
        depth: :attr:`Depth.depth`

    Attributes:
        depth: Matrix containing the distance (in meters) from the camera plane to the world within that pixel in the
            image

    """

    depth: np.ndarray

    def __sizeof__(self):
        return self.depth.nbytes
