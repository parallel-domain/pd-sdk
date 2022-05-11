from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class SemanticSegmentation3D(Annotation):
    """Represents a 3D Semantic Segmentation mask for a point cloud.

    Args:
        class_ids: :attr:`~.SemanticSegmentation3D.class_ids`

    Attributes:
        class_ids: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the class ID for each point as `int`.
    """

    class_ids: np.ndarray

    def __sizeof__(self):
        return self.class_ids.nbytes
