from collections import UserDict
from dataclasses import dataclass
from typing import Dict, Mapping, Type, TypeVar

import numpy as np

from paralleldomain.model.annotation.common import Annotation

T = TypeVar("T")


class CustomAnnotation3D(UserDict, Annotation):  # TODO: Move into separate file
    def __init__(self, __dict: Mapping[str, np.ndarray]):
        super().__init__()
        for key, item in __dict.items():
            if isinstance(key, str) and isinstance(item, np.ndarray):
                super().__setitem__(key=key, item=item)
            else:
                raise TypeError("`key` must be of type `str` and `item` of type `np.ndarray`")

    def __getitem__(self, key: str) -> np.ndarray:
        return super().__getitem__(key=key)

    def __setitem__(self, key: str, item: np.ndarray) -> None:
        if isinstance(key, str) and isinstance(item, np.ndarray):
            super().__setitem__(key=key, item=item)
        else:
            raise TypeError("`key` must be of type `str` and `item` of type `np.ndarray`")

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, np.ndarray]) -> T:
        return CustomAnnotation3D(data)


class CustomMaterialProperties3D(CustomAnnotation3D):
    ...


@dataclass
class MaterialProperties3D(Annotation):
    """Represents a 3D Material Segmentation mask for a point cloud.

    Args:
        material_ids: :attr:`~.MaterialProperties3D.material_ids`

    Attributes:
        material_ids: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material ID for each point as `int`.
    """

    material_ids: np.ndarray
    custom_data: CustomMaterialProperties3D

    def __sizeof__(self):
        return self.material_ids.nbytes + sum([v.nbytes for _, v in self.custom_data.items()])
