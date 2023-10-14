from dataclasses import dataclass
from sys import getsizeof
from typing import List, Optional, Protocol

import numpy as np

from paralleldomain.model.annotation.common import Annotation


class PointCachePointsDecoderProtocol(Protocol):
    """
    Base class for decoding Point Cache Points
    """

    def get_points_xyz(self) -> Optional[np.ndarray]:
        """
        Base implementation of method that extracts xyz positions of points within the point cache

        Returns:
            Array of xyz positions of each point within the point cache
        """
        pass

    def get_points_normals(self) -> Optional[np.ndarray]:
        """
        Base implementation of method that extracts surface normals of points within the point cache

        Returns:
            Array of vector components of the surface normals of each point within the point cache
        """
        pass


class PointCacheComponent:
    """
    A component of a point cache, which contains a subset of points with associated xyz positions and surface normals

    Args:
        component_name: :attr:`PointCacheComponent.component_name`
        points_decoder: An implementation of :obj:`PointCachePointsDecoderProtocol` which defines how the xyz positions
            and surface normals of the points are extracted

    Attributes:
        component_name: The name of the component
    """

    def __init__(self, component_name: str, points_decoder: PointCachePointsDecoderProtocol):
        self._points_decoder = points_decoder
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
        """Returns the xyz positions of the points within the :obj:`PointCacheComponent`"""
        if self._points is None:
            self._points = self._points_decoder.get_points_xyz()
        return self._points

    @property
    def normals(self) -> Optional[np.ndarray]:
        """Returns the surface normals of the points within the :obj:`PointCacheComponent`"""
        if self._normals is None:
            self._normals = self._points_decoder.get_points_normals()
        return self._normals


@dataclass
class PointCache:
    """
    A collection of :obj:`PointCacheComponents` which make up the total point cache of a particular object

    Args:
        instance_id: :attr:`PointCache.instance_id`
        components: :attr:`PointCache.components`

    Attributes:
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation type
        components: List of point cache components which make up a collection of points which is the total point cache
            of an object
    """

    instance_id: int
    components: List[PointCacheComponent]

    def __sizeof__(self):
        return sum([getsizeof(a) for a in self.components])


@dataclass
class PointCaches(Annotation):
    """
    A list of :obj:`PointCache` objects for various objects

    Args:
        caches: :attr:`PointCaches.caches`

    Attributes:
        caches: A list of :obj:`PointCache` objects which are each the point cache of a distinct objects
    """

    caches: List[PointCache]

    def __sizeof__(self):
        return sum([getsizeof(a) for a in self.caches])
