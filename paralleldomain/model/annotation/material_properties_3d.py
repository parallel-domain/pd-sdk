from dataclasses import dataclass
from typing import Optional, TypeVar

import numpy as np

from paralleldomain.model.annotation.common import Annotation

T = TypeVar("T")


@dataclass
class MaterialProperties3D(Annotation):
    """Represents a 3D Material Segmentation mask for a point cloud.

    Args:
        material_ids: :attr:`~.MaterialProperties3D.material_ids`
        metallic: :attr:`~.MaterialProperties3D.metallic`
        specular: :attr:`~.MaterialProperties3D.specular`
        emissive: :attr:`~.MaterialProperties3D.emissive`
        opacity: :attr:`~.MaterialProperties3D.opacity`
        flags: :attr:`~.MaterialProperties3D.flags`

    Attributes:
        material_ids: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material ID for each point as `int`.
        roughness: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material's roughess value for each point as `float`.
        specular: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material's specular value for each point as `float`.
        emissive: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material's emissive value for each point as `float`.
        opacity: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the material's opacity value for each point as `float`.
        flag: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains special flags for each point encoded as `float`.
    """

    material_ids: np.ndarray

    roughness: Optional[np.ndarray] = None
    metallic: Optional[np.ndarray] = None
    specular: Optional[np.ndarray] = None
    emissive: Optional[np.ndarray] = None
    opacity: Optional[np.ndarray] = None
    flags: Optional[np.ndarray] = None

    def __sizeof__(self):
        return (
            self.material_ids.nbytes
            + (self.roughness.nbytes if self.roughness is not None else 0)
            + (self.metallic.nbytes if self.metallic is not None else 0)
            + (self.specular.nbytes if self.specular is not None else 0)
            + (self.emissive.nbytes if self.emissive is not None else 0)
            + (self.opacity.nbytes if self.opacity is not None else 0)
            + (self.flags.nbytes if self.flags is not None else 0)
        )
