from dataclasses import dataclass
from typing import List

import numpy as np

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.utilities.mask import boolean_mask_by_value, boolean_mask_by_values, encode_int32_as_rgb8


@dataclass
class InstanceSegmentation2D(Annotation):
    """
    Represents a 2D Instance Segmentation mask for a camera image.

    Args:
        instance_ids: :attr:`InstanceSegmentation2D.instance_ids`

    Attributes:
        instance_ids: Matrix identical in size to the image where the third axis contains the instance ID for each
            pixel as `int`.
    """

    instance_ids: np.ndarray

    def get_instance(self, instance_id: int) -> np.ndarray:
        """Returns a `bool` mask where instance is present.

        Args:
            instance_id: ID of instance to be masked

        Returns:
            Mask of same shape as :py:attr:`~instance_ids` and `bool` values.
            `True` where pixel matches instance, `False` where it doesn't.
        """
        return boolean_mask_by_value(mask=self.instance_ids, value=instance_id)

    def get_instances(self, instance_ids: List[int]) -> np.ndarray:
        """Returns a `bool` mask where instances are present.

        Args:
            instance_ids: IDs of instances to be masked

        Returns:
            Mask of same shape as `class_ids` and `bool` values.
            `True` where pixel matches one of the instances, `False` where it doesn't.
        """
        return boolean_mask_by_values(mask=self.instance_ids, values=instance_ids)

    def __sizeof__(self):
        return self.instance_ids.nbytes

    @property
    def rgb_encoded(self) -> np.ndarray:
        """
        Outputs :attr:`InstanceSegmentation.instance_ids`
        mask as RGB matrix with shape `(H x W x 3)`,
        with `R` being the lowest and `B` being the highest 8 bit."""
        return encode_int32_as_rgb8(mask=self.instance_ids)

    def __post_init__(self):
        if len(self.instance_ids.shape) != 3:
            raise ValueError("Instance Segmentation instance_ids have to have shape (H x W x 1)")
        if self.instance_ids.dtype != int:
            raise ValueError(
                f"Instance Segmentation instance_ids has to contain only integers but has {self.instance_ids.dtype}!"
            )
        if self.instance_ids.shape[2] != 1:
            raise ValueError("Instance Segmentation instance_ids has to have only 1 channel!")
