from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class Albedo2D(Annotation):
    """
    Contains the base color of each pixel per in the corresponding camera image.

    Args:
        color: :attr:`Albedo2D.color`

    Attributes:
        color: Array containing the base color in RGB format for each pixel of an image, prior to lighting affects being
            applied to the scenario

    """

    color: np.ndarray

    def __sizeof__(self):
        return self.color.nbytes
