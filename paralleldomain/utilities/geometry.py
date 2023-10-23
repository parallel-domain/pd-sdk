import math
import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from more_itertools import pairwise

from paralleldomain.utilities.transformation import Transformation


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
        `True` if point in polygon, otherwise `False`
    """
    polygon = np.asarray(polygon).astype(np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(
            "Expected np.ndarray of shape (N X 2) for `polygon`, where N is  number of points >= 3. Received"
            f" {polygon.shape}."
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
    """Takes a list of 2d points and returns their convex hull

    Args:
        points_2d: n x 2 array of the 2d points we wish to find the convex hull of
        closed: True if the returned convex hull should be closed (first point is same as last point on hull).
            False if returned convex hull should be open

    Returns:
        n x 2 array of the 2d points defining the convex hull
    """
    convex_hull = cv2.convexHull(points=points_2d.reshape((1, -1, 2))).reshape(-1, 2)

    if closed:
        return np.vstack([convex_hull, convex_hull[0]])
    else:
        return convex_hull


def convex_hull_2d_as_mask(points_image_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    """Takes a list of 2d points and returns their convex hull in the form of a filled in mask

    Args:
        points_image_2d: n x 2 array of the 2d points (in image space) we wish to find the convex hull of
        width: The width of the image space on which the points exist and the width of the returned image containing the
            convex hull mask
        height: The height of the image space on which the points exist and the height of the returned image containing
            the convex hull mask

    Returns:
        height x width array containing the mask of the calculated convex hull
    """
    convex_hull = convex_hull_2d(points_2d=points_image_2d, closed=True)
    mask = np.zeros(shape=(height, width)).astype(np.uint8)
    convex_hull_mask = cv2.fillPoly(
        img=mask,
        pts=convex_hull.reshape((1, -1, 2)).astype(int),
        color=255,
    ).astype(bool)

    return convex_hull_mask


def calculate_triangle_area(triangles: np.ndarray) -> np.ndarray:
    """
    Calculates sizes of an array of triangles defined by their three verticies
    Args:
        triangles: n x 3 x 2 array containing the 3 vertices of each triangle

    Returns:
        n x 1 array of triangle sizes
    """
    a_b_c = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2)
    semi_perimeter = np.sum(a_b_c, axis=1) / 2

    # Use Heron's formula to calculate the sizes of the triangles
    return np.sqrt(
        semi_perimeter
        * (semi_perimeter - a_b_c[:, 0])
        * (semi_perimeter - a_b_c[:, 1])
        * (semi_perimeter - a_b_c[:, 2])
    )


def random_point_in_triangle(triangle: np.ndarray, random_seed: int) -> np.ndarray:
    """
    Calculates a random point in a given triangle. Ensuring that the random points are roughly uniformly distributed
    Args:
        triangle: 3 x 2 array containing the x,y position of the triangle's verticies
        random_seed:
    Returns:
        2 x 1 array of x,y position of randomly selected point in each triangle
    """

    random_state = random.Random(random_seed)

    # We use Random instead of np.random since it produces more uniform results
    # triangle = triangle.tolist()
    random_state.shuffle(triangle.tolist())
    triangle = np.array(triangle)

    random_vals = np.array([random_state.uniform(0, 1), random_state.uniform(0, 1), random_state.uniform(0, 1)])
    constants = random_vals / np.sum(random_vals)

    return np.dot(constants, triangle)


def _is_ear_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Function to determine whether a given vertex in a triangle is an ear vertex
    Args:
        vertices: n x 2 array containing the 2d coordinates of the verticies of the polygon

    Returns:
        n x 1 array with each element True if corresponding vertex in vertices is an ear vertex
    """
    num_vertices = len(vertices)

    shift_forward_vertices = np.roll(vertices, 1, axis=0)
    shift_backward_vertices = np.roll(vertices, -1, axis=0)

    triangles_mat = np.hstack((shift_forward_vertices, vertices, shift_backward_vertices))
    triangles_mat = triangles_mat.reshape(num_vertices, 3, 2)

    edges_mat = np.roll(triangles_mat, -1, axis=1) - triangles_mat

    angles_mat = np.cross(edges_mat[:, :-1], edges_mat[:, 1:])

    interior_angles_mat = np.arctan2(np.abs(angles_mat), np.sum(edges_mat[:, :-1] * edges_mat[:, 1:], axis=-1))

    return np.all(interior_angles_mat <= np.pi, axis=1)


def decompose_polygon_into_triangles(vertices: np.ndarray) -> np.ndarray:
    """Takes an area defined by a 2D polygon and decomposes it into triangles using the Ear Clipping Method.

    Args:
        vertices: n x 2 array containing the 2d coordinates of the vertices of the polygon

    Returns:
        n x 3 x 2 array containing the 3 vertices of each n triangle
    """
    num_vertices = len(vertices)

    is_ear = _is_ear_vertices(vertices=vertices)

    triangles = np.zeros((num_vertices - 2, 3, 2))
    indices = np.arange(num_vertices)

    found_triangles = 0

    while num_vertices >= 3:
        ear_index = np.argmax(is_ear)

        # triangles.append(
        #     (vertices[(ear_index - 1) % num_vertices], vertices[ear_index], vertices[(ear_index + 1) % num_vertices])
        # )

        triangles[found_triangles] = np.vstack(
            (vertices[(ear_index - 1) % num_vertices], vertices[ear_index], vertices[(ear_index + 1) % num_vertices])
        )
        found_triangles += 1

        # Remove the ear vertex from the polygon
        vertices = np.delete(vertices, ear_index, axis=0)
        indices = np.delete(indices, ear_index)
        is_ear = np.delete(is_ear, ear_index)
        num_vertices -= 1

        if num_vertices >= 3:
            prev_idx = (ear_index - 1) % num_vertices
            next_idx = ear_index % num_vertices
            prev_vertex = vertices[prev_idx]
            current_vertex = vertices[next_idx]
            next_vertex = vertices[(next_idx + 1) % num_vertices]

            triangle_to_test = np.vstack((prev_vertex, current_vertex, next_vertex))

            is_ear[prev_idx] = _is_ear_vertices(triangle_to_test)[0]
            is_ear[next_idx] = _is_ear_vertices(triangle_to_test)[-1]

    return triangles


def random_point_within_2d_polygon(edge_2d: np.ndarray, random_seed: int, num_points: int = 1) -> np.ndarray:
    """
    Returns a random point within a polygon.  Note that this returns a z value of 0
    Args:
        edge_2d: n x 2 array containing the 2d coordinates of the verticies of the polygon
        random_seed: random seed - will be overridden if randomize_point_selection is set to True
        num_points: Number of points to spawn
    Returns:
        num_points x 2 array containing the x,y positions of the point(s)
    """
    decomposed_triangles = decompose_polygon_into_triangles(edge_2d)
    triangle_weights = calculate_triangle_area(decomposed_triangles)

    points = np.zeros((num_points, 2))
    for i in range(num_points):
        seed = random_seed + i
        random_state = random.Random(seed)

        points[i, :] = random_point_in_triangle(
            triangle=np.array(random_state.choices(decomposed_triangles, weights=triangle_weights))[0],
            random_seed=seed,
        )

    return points
