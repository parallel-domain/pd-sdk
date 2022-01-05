from typing import List, Tuple, Union

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
    supporting_points_indices: List[int] = None,
    approximation_error: float = 0.1,
) -> np.ndarray:
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
