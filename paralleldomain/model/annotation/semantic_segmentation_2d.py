from dataclasses import dataclass
from typing import List

import numpy as np

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.utilities.mask import boolean_mask_by_value, boolean_mask_by_values, encode_int32_as_rgb8


@dataclass
class SemanticSegmentation2D(Annotation):
    """Represents a 2D Semantic Segmentation mask for a camera image.

    Args:
        class_ids: :attr:`paralleldomain.model.annotation.semantic_segmentation_2d.SemanticSegmentation2D.class_ids`

    Attributes:
        class_ids: Matrix of shape `(H x W x 1)`, where `H` is height and `W` is width of corresponding camera image.
            The third axis contains the class ID for each pixel as `int`.
    """

    class_ids: np.ndarray

    def get_class_mask(self, class_id: int) -> np.ndarray:
        """Returns a `bool` mask where class is present.

        Args:
            class_id: ID of class to be masked

        Returns:
            Mask of same shape as :py:attr:`~class_ids` and `bool` values.
            `True` where pixel matches class, `False` where it doesn't.
        """
        return boolean_mask_by_value(mask=self.class_ids, value=class_id)

    def get_classes_mask(self, class_ids: List[int]) -> np.ndarray:
        """Returns a `bool` mask where classes are present.

        Args:
            class_ids: IDs of classes to be masked

        Returns:
            Mask of same shape as `class_ids` and `bool` values.
            `True` where pixel matches one of the classes, `False` where it doesn't.
        """
        return boolean_mask_by_values(mask=self.class_ids, values=class_ids)

    @property
    def rgb_encoded(self) -> np.ndarray:
        """Outputs :attr:`paralleldomain.model.annotation.semantic_segmentation_2d.SemanticSegmentation.class_ids` mask
        as RGB-encoded image matrix with shape `(H x W x 3)`,
        with `R` (index: 0) being the lowest and `B` (index: 2) being the highest 8 bit."""
        return encode_int32_as_rgb8(mask=self.class_ids)

    def __post_init__(self):
        if len(self.class_ids.shape) != 3:
            raise ValueError("Semantic Segmentation class_ids have to have shape (H x W x 1)")
        if self.class_ids.dtype != int:
            raise ValueError(
                f"Semantic Segmentation class_ids has to contain only integers but has {self.class_ids.dtype}!"
            )
        if self.class_ids.shape[2] != 1:
            raise ValueError("Semantic Segmentation class_ids has to have only 1 channel!")

    def __sizeof__(self):
        return self.class_ids.nbytes
