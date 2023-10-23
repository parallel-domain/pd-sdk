from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class SurfaceNormals2D(Annotation):
    """Represents a mask of surface normals for a camera image.

    Args:
        normals: :attr:`SurfaceNormals2D.normals`

    Attributes:
        normals: Matrix of shape `(H X W x 3)`, where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the x, y and z normal direction of the surface sampled by the pixel
            in the camera coordinate system.
    """

    normals: np.ndarray

    def __sizeof__(self):
        return self.normals.nbytes
