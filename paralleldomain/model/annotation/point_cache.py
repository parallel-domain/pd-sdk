from dataclasses import dataclass
from sys import getsizeof
from typing import List

import numpy as np

try:
    from typing import Optional, Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.utilities.transformation import Transformation


class PointCachePointsDecoderProtocol(Protocol):
    def get_points_xyz(self) -> Optional[np.ndarray]:
        pass

    def get_points_normals(self) -> Optional[np.ndarray]:
        pass


class PointCacheComponent:
    def __init__(self, component_name: str, pose: Transformation, points_decoder: PointCachePointsDecoderProtocol):
        self._points_decoder = points_decoder
        self.pose = pose
        self.component_name = component_name
        self._points = None
        self._normals = None

    def __sizeof__(self):
        size = 0
        if self._points is not None:
            size += self._points.nbytes
        if self._normals is not None:
            size += self._normals.nbytes
        return size

    @property
    def points(self) -> Optional[np.ndarray]:
        if self._points is None:
            self._points = self._points_decoder.get_points_xyz()
        return self._points

    @property
    def normals(self) -> Optional[np.ndarray]:
        if self._normals is None:
            self._normals = self._points_decoder.get_points_normals()
        return self._normals


@dataclass
class PointCache:
    """Represents a 3D Bounding Box geometry.

    Args:
        instance_id: :attr:`~.PointCache.instance_id`
        components: :attr:`~.PointCache.components`

    Attributes:
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`BoundingBox3D` or :obj:`InstanceSegmentation3D`.
        components: The point cache component containing the point cloud.
    """

    instance_id: int
    components: List[PointCacheComponent]

    def __sizeof__(self):
        return sum([getsizeof(a) for a in self.components])


@dataclass
class PointCaches(Annotation):
    """Represents a 3D Bounding Box geometry.

    Args:
        instance_id: :attr:`~.PointCache.instance_id`
        components: :attr:`~.PointCache.components`

    Attributes:
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`BoundingBox3D` or :obj:`InstanceSegmentation3D`.
        components: The point cache component containing the point cloud.
    """

    caches: List[PointCache]

    def __sizeof__(self):
        return sum([getsizeof(a) for a in self.caches])
