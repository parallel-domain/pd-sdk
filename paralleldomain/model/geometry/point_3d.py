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

    def transform(self, tf: Transformation) -> "Point3DBaseGeometry[T]":
        tf_point = (tf @ np.array([self.x, self.y, self.z, 1]))[:3]
        return Point3DBaseGeometry[T](
            x=self._ensure_type(tf_point[0]), y=self._ensure_type(tf_point[1]), z=self._ensure_type(tf_point[2])
        )

    def _ensure_type(self, value: Union[int, float]) -> T:
        try:
            actual_type = typing.get_args(self.__orig_class__)[0]
            if isinstance(actual_type, type):
                return actual_type(value)
            else:
                return type(self.x)(value)
        except AttributeError:
            return type(self.x)(value)

    def __add__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x + other.x),
                y=self._ensure_type(self.y + other.y),
                z=self._ensure_type(self.z + other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x + other),
                y=self._ensure_type(self.y + other),
                z=self._ensure_type(self.z + other),
            )
        else:
            raise ValueError(f"Unsupported value {other} of type {type(other)}!")

    def __radd__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x + other.x),
                y=self._ensure_type(self.y + other.y),
                z=self._ensure_type(self.z + other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x + other),
                y=self._ensure_type(self.y + other),
                z=self._ensure_type(self.z + other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __sub__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x - other.x),
                y=self._ensure_type(self.y - other.y),
                z=self._ensure_type(self.z - other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x - other),
                y=self._ensure_type(self.y - other),
                zy=self._ensure_type(self.z - other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rsub__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x - other.x),
                y=self._ensure_type(self.y - other.y),
                z=self._ensure_type(self.z - other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x - other),
                y=self._ensure_type(self.y - other),
                z=self._ensure_type(self.z - other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __mul__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x * other.x),
                y=self._ensure_type(self.y * other.y),
                z=self._ensure_type(self.z * other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x * other),
                y=self._ensure_type(self.y * other),
                z=self._ensure_type(self.z * other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rmul__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x * other.x),
                y=self._ensure_type(self.y * other.y),
                z=self._ensure_type(self.z * other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x * other),
                y=self._ensure_type(self.y * other),
                z=self._ensure_type(self.z * other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __div__(self, other):
        if isinstance(other, Point3DBaseGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x / other.x),
                y=self._ensure_type(self.y / other.y),
                z=self._ensure_type(self.z / other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x / other),
                y=self._ensure_type(self.y / other),
                z=self._ensure_type(self.z / other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rdiv__(self, other):
        if isinstance(other, Point3DGeometry):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x / other.x),
                y=self._ensure_type(self.y / other.y),
                z=self._ensure_type(self.z / other.z),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return Point3DBaseGeometry[T](
                x=self._ensure_type(self.x / other),
                y=self._ensure_type(self.y / other),
                z=self._ensure_type(self.z / other),
            )
        else:
            raise ValueError(f"Unsupported value {other}!")

    @classmethod
    def from_numpy(cls, point: np.ndarray):
        point = point.reshape(3)
        return cls(x=point[0], y=point[1], z=point[2])


class Point3DGeometry(Point3DBaseGeometry[float]):
    pass
