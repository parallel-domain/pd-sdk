from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class InstanceSegmentation3D(Annotation):
    """Represents a 3D Instance Segmentation mask for a point cloud.

    Args:
        instance_ids: :attr:`~.InstanceSegmentation3D.instance_ids`

    Attributes:
        instance_ids: 2D Matrix of size `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the instance ID for each point as `int`.
    """

    instance_ids: np.ndarray

    def __sizeof__(self):
        return self.instance_ids.nbytes
