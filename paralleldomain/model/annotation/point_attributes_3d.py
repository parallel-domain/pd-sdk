from dataclasses import dataclass
from sys import getsizeof
from typing import List

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class PointAttributes3D(Annotation):
    """Represents a mask for a point cloud .

    Args:
        attributes: :attr:`~.PointAttributes3D.properties`
        channel_names: :attr:`~.PointAttributes3D.channel_names`

    Attributes:
        attributes: 2D Matrix of size `(N x M)`, where `N` is the length of the corresponding point cloud.
            The second axis contains an arbitrary amount of channels with attributes.
        channel_names: List of length M containing a name that describes the content
            of each channel (like 'roughness' or 'is_wet").
    """

    attributes: np.ndarray
    channel_names: List[str]

    def __sizeof__(self):
        return self.attributes.nbytes + getsizeof(self.channel_names)
