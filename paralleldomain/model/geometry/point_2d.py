import typing
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

import numpy as np

T = TypeVar("T", int, float)


@dataclass
class Point2DBaseGeometry(Generic[T]):
    """Represents a 2D Point.

    Args:
        x: :attr:`~.Point2D.x`
        y: :attr:`~.Point2D.y`
        class_id: :attr:`~.Point2D.class_id`
        instance_id: :attr:`~.Point2D.instance_id`
        attributes: :attr:`~.Point2D.attributes`

    Attributes:
        x: coordinate along x-axis in image pixels
        y: coordinate along y-axis in image pixels
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    x: T
    y: T

    def to_numpy(self):
        """Returns the coordinates as a numpy array with shape (1 x 2)."""
        return np.array([[self.x, self.y]])

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
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x + other.x), y=self._ensure_type(self.y + other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x + other), y=self._ensure_type(self.y + other))
        else:
            raise ValueError(f"Unsupported value {other} of type {type(other)}!")

    def __radd__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x + other.x), y=self._ensure_type(self.y + other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x + other), y=self._ensure_type(self.y + other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __sub__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x - other.x), y=self._ensure_type(self.y - other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x - other), y=self._ensure_type(self.y - other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rsub__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x - other.x), y=self._ensure_type(self.y - other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x - other), y=self._ensure_type(self.y - other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __mul__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x * other.x), y=self._ensure_type(self.y * other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x * other), y=self._ensure_type(self.y * other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rmul__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x * other.x), y=self._ensure_type(self.y * other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x * other), y=self._ensure_type(self.y * other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __div__(self, other):
        if isinstance(other, Point2DBaseGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x / other.x), y=self._ensure_type(self.y / other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x / other), y=self._ensure_type(self.y / other))
        else:
            raise ValueError(f"Unsupported value {other}!")

    def __rdiv__(self, other):
        if isinstance(other, Point2DGeometry):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x / other.x), y=self._ensure_type(self.y / other.y))
        elif isinstance(other, int) or isinstance(other, float):
            return Point2DBaseGeometry[T](x=self._ensure_type(self.x / other), y=self._ensure_type(self.y / other))
        else:
            raise ValueError(f"Unsupported value {other}!")


class Point2DGeometry(Point2DBaseGeometry[int]):
    pass
