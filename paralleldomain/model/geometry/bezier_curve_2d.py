from typing import Generic, List, TypeVar, Union

import numpy as np
from bezier import Curve

from paralleldomain.model.geometry.point_2d import Point2DBaseGeometry
from paralleldomain.model.geometry.polyline_2d import Polyline2DBaseGeometry

T = TypeVar("T", int, float)


class BezierCurve2DBaseGeometry(Generic[T]):
    def __init__(self, control_points: Union[List[Point2DBaseGeometry[T]], np.ndarray]):
        if isinstance(control_points, list):
            if not control_points:
                raise ValueError("The list of control points must have at least 2 entries.")
            unique_types = {type(x) for x in control_points}
            if len(unique_types) > 1:
                raise ValueError("The list of control points must have the same type.")
            if not isinstance(control_points[0], Point2DBaseGeometry):
                raise TypeError(
                    f"The list contains elements of type {type(control_points[0])} instead of `Point2DBaseGeometry`."
                )

            control_points = np.vstack([cp.to_numpy() for cp in control_points])

        if control_points.shape[1] != 2:
            raise ValueError(
                "The control points can only be 2-dimensional points. The provided points have dimension"
                f" {control_points.shape[1]}."
            )

        self._bezier_curve = Curve.from_nodes(nodes=control_points.T)

    @property
    def length(self) -> float:
        """Returns the length of the curve."""
        return self._bezier_curve.length

    @property
    def control_points(self) -> np.ndarray:
        """Returns the control points of the curve."""
        return self._bezier_curve.nodes.T

    def evaluate_at(self, value: float) -> Point2DBaseGeometry[T]:
        """Evaluates the curve at the given relative float value `[0.0, 0.1]`.

        Args:
            value: A relative float value `[0.0, 0.1]` at which to evaluate the curve.

        Returns: The absolute point on the curve at the given value.
        """
        return Point2DBaseGeometry[T].from_numpy(self._bezier_curve.evaluate(s=value))

    def tangent_at(self, value: float, normalize: bool = True) -> Point2DBaseGeometry[T]:
        """Calculates the tangent of the curve at the given value using a hodograph.

        Args:
            value: A relative float value `[0.0, 0.1]` at which to calculate the tangent.
            normalize: Whether to normalize the tangent vector. Defaults to `True`.

        Returns: The tangent directional 2D vector of the curve at the given relative value.

        """
        tangent = self._bezier_curve.evaluate_hodograph(s=value).T
        if normalize:
            tangent = tangent / np.linalg.norm(tangent)

        return Point2DBaseGeometry[T].from_numpy(tangent)

    def as_polyline(self, num_points: int = 100) -> Polyline2DBaseGeometry[float]:
        """Returns the curve as a polyline sampled with `num_points` points and point type `float`."""
        sample_relative = np.linspace(start=0.0, stop=1.0, num=num_points)
        sample_absolute = self._bezier_curve.evaluate_multi(s_vals=sample_relative)

        return Polyline2DBaseGeometry[float].from_numpy(sample_absolute.T)


class BezierCurve2DGeometry(BezierCurve2DBaseGeometry[float]):
    pass
