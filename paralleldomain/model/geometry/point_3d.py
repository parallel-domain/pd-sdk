import typing
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

import numpy as np

from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T", int, float)


@dataclass
class Point3DBaseGeometry(Generic[T]):
    """Represents a 3D Point.

    Args:
        x: :attr:`~.Point3DBaseGeometry.x`
        y: :attr:`~.Point3DBaseGeometry.y`
        z: :attr:`~.Point3DBaseGeometry.z`
        class_id: :attr:`~.Point3DBaseGeometry.class_id`
        instance_id: :attr:`~.Point3DBaseGeometry.instance_id`
        attributes: :attr:`~.Point3DBaseGeometry.attributes`

    Attributes:
        x: coordinate along x-axis in sensor coordinates
        y: coordinate along y-axis in sensor coordinates
        z: coordinate along z-axis in sensor coordinates
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    x: T
    y: T
    z: T

    def to_numpy(self):
        """Returns the coordinates as a numpy array with shape (1 x 3)."""
        return np.array([[self.x, self.y, self.z]])

    def transform(self, tf: Transformation) -> "Point3DGeometry":
        tf_point = (tf @ np.array([self.x, self.y, self.z, 1]))[:3]
        return Point3DBaseGeometry[T](
            x=self._ensure_type(tf_point[0]), y=self._ensure_type(tf_point[1]), z=self._ensure_type(tf_point[2])
        )

    def _ensure_type(self, value: Union[int, float]) -> T:
        actual_type = typing.get_args(self.__orig_class__)[0]
        return actual_type(value)

    @classmethod
    def from_numpy(cls, point: np.ndarray):
        pt = point.reshape(-3)
        return cls(x=pt[0], y=pt[1], z=pt[2])


class Point3DGeometry(Point3DBaseGeometry[float]):
    pass
