from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class Depth(Annotation):
    """Represents a Depth mask for a camera image.



    Args:
        depth: :attr:`paralleldomain.model.annotation.depth.Depth.depth`

    Attributes:
        depth: Matrix of shape `(H X W x 1)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the depth distance for each pixel as `float` in meter.

    """

    depth: np.ndarray

    def __sizeof__(self):
        return self.depth.nbytes
