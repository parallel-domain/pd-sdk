from dataclasses import dataclass
from sys import getsizeof

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class Depth(Annotation):
    """Represents a Depth mask for a camera image.



    Args:
        depth: :attr:`~.Depth.depth`

    Attributes:
        depth: Matrix of shape `(H X W x 1)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the depth distance for each pixel as `int` in meter.

    """

    depth: np.ndarray

    def __sizeof__(self):
        return self.depth.nbytes
