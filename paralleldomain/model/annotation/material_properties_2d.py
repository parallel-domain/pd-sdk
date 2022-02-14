from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class MaterialProperties2D(Annotation):
    """Contains the Metallic/Specular roughness of each pixel per in the corresponding camera image.



    Args:
        roughness: :attr:`~.MetallicSpecularRoughness.roughness:`

    Attributes:
        roughness:: Matrix of shape `(H X W x 3)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the Metallic/Specular roughness of each pixel.

    """

    roughness: np.ndarray

    def __sizeof__(self):
        return self.roughness.nbytes
