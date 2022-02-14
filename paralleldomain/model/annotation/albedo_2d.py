from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class Albedo2D(Annotation):
    """Contains the base color of each pixel per in the corresponding camera image.



    Args:
        color: :attr:`~.BaseColor.color:`

    Attributes:
        color:: Matrix of shape `(H X W x 3)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the base color of each pixel before lighting takes place.

    """

    color: np.ndarray

    def __sizeof__(self):
        return self.color.nbytes
