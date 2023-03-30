from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from more_itertools import pairwise


def interpolate_points(points: np.ndarray, num_points: int, flatten_result: bool = True) -> np.ndarray:
    """Takes a list of points and interpolates a number of additional points inbetween each point pair.

    Args:
        points: Array of consecutive points
        num_points: Every start point per pair will result in `num_points` points after interpolation.
        flatten_result: If `False`
            will return interpolated points, where first axis groups by each input point pair; else returns flat list
            of all points. Default: `True`

    Returns:
        Array of points with linearly interpolated values inbetween.
    """
    if points.ndim != 2:
        raise ValueError(
            f"""Expected np.ndarray of shape (N X M) for `points`, where N is
                number of points and M number of dimensions. Received {points.shape}."""
        )
    if num_points < 2:
        raise ValueError(f"`num_points` must be at least 2, received {num_points}")

    factors_lin = np.linspace(0, 1, num_points, endpoint=False)
    factors = np.stack([1 - factors_lin, factors_lin], axis=-1)
    point_pairs = np.stack([points[:-1], points[1:]], axis=1)

    point_pairs_interp = factors @ point_pairs

    return point_pairs_interp.reshape(-1, points.shape[1]) if flatten_result else point_pairs_interp


def is_point_in_polygon_2d(
    polygon: Union[np.ndarray, List[Union[List[float], Tuple[float, float]]]],
    point: Union[np.ndarray, List[float], Tuple[float, float]],
    include_edge: bool = True,
) -> bool:
    """Checks if a point lies inside a polygon shape.

    Args:
        polygon: Array of consecutive points that form a polygon
        point: 2D point coordinates to be tested if they lie in the specified polygon
        include_edge: If point is considered inside if lying on the edge or not. Default: `True`

    Returns:
        `True` if point in polygon, otherwise `False,
    """
    polygon = np.asarray(polygon).astype(np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(
            f"""Expected np.ndarray of shape (N X 2) for `polygon`, where N is
                number of points >= 3. Received {polygon.shape}."""
        )

    if isinstance(point, np.ndarray):
        point = point.tolist()
    point = tuple(map(float, point))
    if len(point) != 2:
        raise ValueError(f"""Expected `points` with length 2. Received {len(point)}.""")

    threshold = 0 if include_edge else 1
    return cv2.pointPolygonTest(contour=polygon, pt=point, measureDist=False) >= threshold


def simplify_polyline_2d(
    polyline: Union[np.ndarray, List[Union[List[float], Tuple[float, float]]]],
    supporting_points_indices: Optional[List[int]] = None,
    approximation_error: float = 0.1,
) -> np.ndarray:
    """Takes a 2D polyline and simplifies its shape while allowing for a specified error.

    Args:
        polyline: 2D Polyline that should be simplified.
        supporting_points_indices: An optional list of vertices of the polyline that need to be kept during
            simplification.
        approximation_error: The maximum error that's allowed to be introduced during simplification.

    Returns:
        A simplified version of the input polyline, or the input polyline if no simplification could be done.
    """
    polyline = np.asarray(polyline)
    if polyline.ndim != 2 or polyline.shape[1] != 2:
        raise ValueError(f"""Expected np.ndarray of shape (N X 2) for `polyline`. Received {polyline.shape}.""")
    if len(polyline) < 3:
        return polyline
    elif supporting_points_indices is None or len(supporting_points_indices) == 0:
        return cv2.approxPolyDP(curve=polyline, epsilon=approximation_error, closed=False).reshape(-1, 2)
    else:
        supporting_points_indices = [0] + supporting_points_indices + [len(polyline)]
        supporting_points_indices = sorted(list(set(supporting_points_indices)))
        supporting_points_pairs = list(pairwise(supporting_points_indices))
        polyline_simplified = np.empty(shape=(0, 2))
        for spp in supporting_points_pairs:
            polyline_sub = polyline[spp[0] : spp[1] + 1].astype(np.float32)
            polyline_sub_simplified = cv2.approxPolyDP(
                curve=polyline_sub, epsilon=approximation_error, closed=False
            ).reshape(-1, 2)[:-1, :]

            polyline_simplified = np.vstack([polyline_simplified, polyline_sub_simplified])
        polyline_simplified = np.vstack([polyline_simplified, polyline[-1, :]])

        return polyline_simplified


def convex_hull_2d(points_2d: np.ndarray, closed: bool = False) -> np.ndarray:
    convex_hull = cv2.convexHull(points=points_2d.reshape((1, -1, 2))).reshape(-1, 2)

    if closed:
        return np.vstack([convex_hull, convex_hull[0]])
    else:
        return convex_hull


def convex_hull_2d_as_mask(points_image_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    convex_hull = convex_hull_2d(points_2d=points_image_2d, closed=True)
    mask = np.zeros(shape=(height, width)).astype(np.uint8)
    convex_hull_mask = cv2.fillPoly(
        img=mask,
        pts=convex_hull.reshape((1, -1, 2)).astype(int),
        color=255,
    ).astype(bool)

    return convex_hull_mask
