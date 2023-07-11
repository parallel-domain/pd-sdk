from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from paralleldomain.constants import CAMERA_MODEL_OPENCV_PINHOLE
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.projection import project_points_2d_to_3d, project_points_3d_to_2d
from paralleldomain.utilities.transformation import Transformation


@dataclass()
class UndistortionOutput:
    input_image: np.ndarray
    empty_image: np.ndarray
    input_mask: np.ndarray
    virtual_camera_intrinsic: np.ndarray
    virtual_camera_to_actual_sensor_in_rdf: Transformation


def undistort_inpainting_input(
    input_image: np.ndarray,
    empty_image: np.ndarray,
    input_mask: np.ndarray,
    depth: np.ndarray,
    camera: CameraSensorFrame,
    context_scale: float,
    context_scale_pad_factor: float,
) -> UndistortionOutput:
    center_x, center_y, _, _ = _get_bounding_box_from_mask(input_mask=input_mask)

    virtual_camera_3d_target = project_points_2d_to_3d(
        k_matrix=camera.intrinsic.camera_matrix,
        camera_model=camera.intrinsic.camera_model,
        distortion_lookup=camera.distortion_lookup,
        distortion_parameters=camera.intrinsic.distortion_parameters,
        points_2d=np.array([[center_x, center_y]]),
        depth=depth,
    )[0]
    RDF_TO_FLU = (CoordinateSystem("RDF") > CoordinateSystem("FLU")).rotation

    target_position_in_sensor_flu = camera.sensor_to_ego.rotation @ RDF_TO_FLU @ virtual_camera_3d_target
    ego_to_virtual_camera = Transformation.look_at(
        target=target_position_in_sensor_flu, coordinate_system="FLU"
    ).inverse
    virtual_camera_to_actual_sensor_in_rdf = CoordinateSystem.change_transformation_coordinate_system(
        transformation=Transformation(quaternion=camera.ego_to_sensor.quaternion) @ ego_to_virtual_camera.inverse,
        transformation_system="FLU",
        target_system="RDF",
    )

    virtual_camera_fov, virtual_camera_resolution = _calculate_virtual_camera_fov_and_resolution(
        input_mask=input_mask,
        camera=camera,
        virtual_camera_to_actual_sensor_in_rdf=virtual_camera_to_actual_sensor_in_rdf,
        context_scale=context_scale,
        depth=depth,
        context_scale_pad_factor=context_scale_pad_factor,
    )
    focal_length = virtual_camera_resolution / (2 * np.tan(np.deg2rad(virtual_camera_fov) / 2))
    virtual_camera_intrinsic = np.array(
        [
            [focal_length, 0, virtual_camera_resolution / 2],
            [0, focal_length, virtual_camera_resolution / 2],
            [0, 0, 1],
        ]
    )

    distorted_2d_coordinates = _create_remapping_lut(
        target_width=virtual_camera_resolution,
        target_height=virtual_camera_resolution,
        camera_to_virtual=False,
        virtual_camera_k_matrix=virtual_camera_intrinsic,
        camera=camera,
        position_3d_transformation=virtual_camera_to_actual_sensor_in_rdf,
        depth=np.ones((virtual_camera_resolution, virtual_camera_resolution, 1)),
    )

    undistorted_input_image = cv2.remap(
        input_image,
        distorted_2d_coordinates[..., 0].astype(np.float32),
        distorted_2d_coordinates[..., 1].astype(np.float32),
        cv2.INTER_LANCZOS4,
    )

    undistorted_input_mask = cv2.remap(
        input_mask,
        distorted_2d_coordinates[..., 0].astype(np.float32),
        distorted_2d_coordinates[..., 1].astype(np.float32),
        cv2.INTER_LANCZOS4,
    )

    undistorted_empty_image = cv2.remap(
        empty_image,
        distorted_2d_coordinates[..., 0].astype(np.float32),
        distorted_2d_coordinates[..., 1].astype(np.float32),
        cv2.INTER_LANCZOS4,
    )

    return UndistortionOutput(
        input_image=undistorted_input_image,
        input_mask=undistorted_input_mask,
        empty_image=undistorted_empty_image,
        virtual_camera_intrinsic=virtual_camera_intrinsic,
        virtual_camera_to_actual_sensor_in_rdf=virtual_camera_to_actual_sensor_in_rdf,
    )


def distort_reactor_output(
    output_image: np.ndarray,
    output_mask: np.ndarray,
    undistortion_data: UndistortionOutput,
    input_image: np.ndarray,
    camera: CameraSensorFrame,
    depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    result_width = input_image.shape[1]
    result_height = input_image.shape[0]

    undistorted_crop_2d_coordinates = _create_remapping_lut(
        target_width=result_width,
        target_height=result_height,
        camera_to_virtual=True,
        virtual_camera_k_matrix=undistortion_data.virtual_camera_intrinsic,
        camera=camera,
        position_3d_transformation=undistortion_data.virtual_camera_to_actual_sensor_in_rdf.inverse,
        depth=depth,
    )
    distorted_image = cv2.remap(
        output_image,
        undistorted_crop_2d_coordinates[..., 0].astype(np.float32),
        undistorted_crop_2d_coordinates[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )

    distorted_mask = cv2.remap(
        output_mask,
        undistorted_crop_2d_coordinates[..., 0].astype(np.float32),
        undistorted_crop_2d_coordinates[..., 1].astype(np.float32),
        cv2.INTER_NEAREST,
    )
    alpha = (distorted_image[..., -1:] == 255).astype(np.float32)
    merged_image = (1 - alpha) * input_image + alpha * distorted_image
    return np.clip(merged_image, 0, 255).astype(np.uint8), distorted_mask.astype(np.uint8)


def _create_remapping_lut(
    target_width: int,
    target_height: int,
    camera_to_virtual: bool,
    virtual_camera_k_matrix: np.ndarray,
    camera: CameraSensorFrame,
    position_3d_transformation: Transformation,
    depth: np.ndarray,
):
    # For each target pixel calculate the position in the source image by going from 2d to 3d to 2d again.
    target_2d_pixel_coordinates = np.empty((target_height, target_width, 2))
    target_2d_pixel_coordinates[..., 0] = np.arange(target_width)[np.newaxis, :]
    target_2d_pixel_coordinates[..., 1] = np.arange(target_height)[:, np.newaxis]
    if camera_to_virtual is False:
        k_matrix = virtual_camera_k_matrix
        camera_model = CAMERA_MODEL_OPENCV_PINHOLE
        distortion_lookup = None
        distortion_parameters = None
    else:
        k_matrix = camera.intrinsic.camera_matrix
        camera_model = camera.intrinsic.camera_model
        distortion_lookup = camera.distortion_lookup
        distortion_parameters = camera.intrinsic.distortion_parameters
    target_3d_coordinates = project_points_2d_to_3d(
        k_matrix=k_matrix,
        camera_model=camera_model,
        distortion_lookup=distortion_lookup,
        distortion_parameters=distortion_parameters,
        points_2d=target_2d_pixel_coordinates.reshape((-1, 2)),
        depth=depth,
    )
    virtual_camera_3d_coordinates = (
        position_3d_transformation.transformation_matrix[:3, :3] @ target_3d_coordinates.T
    ).T

    if camera_to_virtual is False:
        k_matrix = camera.intrinsic.camera_matrix
        camera_model = camera.intrinsic.camera_model
        distortion_lookup = camera.distortion_lookup
        distortion_parameters = camera.intrinsic.distortion_parameters
    else:
        k_matrix = virtual_camera_k_matrix
        camera_model = CAMERA_MODEL_OPENCV_PINHOLE
        distortion_lookup = None
        distortion_parameters = None
    source_2d_coordinates = project_points_3d_to_2d(
        k_matrix=k_matrix,
        camera_model=camera_model,
        distortion_lookup=distortion_lookup,
        distortion_parameters=distortion_parameters,
        points_3d=virtual_camera_3d_coordinates,
    )
    return source_2d_coordinates.reshape((target_height, target_width, 2))


def _calculate_virtual_camera_fov_and_resolution(
    input_mask: np.ndarray,
    camera: CameraSensorFrame,
    virtual_camera_to_actual_sensor_in_rdf: Transformation,
    depth: np.ndarray,
    context_scale: float,
    context_scale_pad_factor: float,
) -> Tuple[float, int]:
    center_x, center_y, height, width = _get_bounding_box_from_mask(input_mask=input_mask)

    context_scaled_width = (context_scale_pad_factor * context_scale) * width
    context_scaled_height = (context_scale_pad_factor * context_scale) * height

    corners = np.clip(
        [
            [center_x - 0.5 * context_scaled_width, center_y - 0.5 * context_scaled_height],
            [center_x + 0.5 * context_scaled_width, center_y + 0.5 * context_scaled_height],
        ],
        [0, 0],
        [depth.shape[1] - 1, depth.shape[0] - 1],
    )
    clipped_width = np.ceil(corners[1, 0] - corners[0, 0]).astype(np.int64)
    clipped_height = np.ceil(corners[1, 1] - corners[0, 1]).astype(np.int64)
    # Problem: You can create images, where not all pixels are covered with values, depth is nan it that case
    # We can't unproject a corner in that area, so we unproject all valid pixels.
    pixel_coordinates = np.empty((clipped_height, clipped_width, 2), dtype=np.int64)
    pixel_coordinates[..., 0] = np.floor(corners[0, 0]) + np.arange(clipped_width)[np.newaxis, :]
    pixel_coordinates[..., 1] = np.floor(corners[0, 1]) + np.arange(clipped_height)[:, np.newaxis]
    pixel_coordinates = pixel_coordinates.reshape((clipped_height * clipped_width, 2))
    pixel_coordinates = pixel_coordinates[
        np.logical_not(np.isnan(depth[pixel_coordinates[:, 1], pixel_coordinates[:, 0]]))[:, 0]
    ]

    # If we have the box corners in 3d camera space we can simply take the maximum of the angles to the cam
    corners_3d = project_points_2d_to_3d(
        k_matrix=camera.intrinsic.camera_matrix,
        camera_model=camera.intrinsic.camera_model,
        distortion_lookup=camera.distortion_lookup,
        distortion_parameters=camera.intrinsic.distortion_parameters,
        points_2d=pixel_coordinates,
        depth=depth,
    )
    corners_3d_in_virtual_cam_rdf = (
        virtual_camera_to_actual_sensor_in_rdf.inverse.transformation_matrix[:3, :3] @ corners_3d.T
    ).T
    # The interpolating depth access still leads to some nans
    corners_3d_in_virtual_cam_rdf = corners_3d_in_virtual_cam_rdf[
        ~np.any(np.isnan(corners_3d_in_virtual_cam_rdf), axis=-1)
    ]
    angles_horizontal = np.rad2deg(
        np.abs(np.arctan2(corners_3d_in_virtual_cam_rdf[:, 0], corners_3d_in_virtual_cam_rdf[:, 2]))
    )
    angles_vertical = np.rad2deg(
        np.abs(np.arctan2(corners_3d_in_virtual_cam_rdf[:, 1], corners_3d_in_virtual_cam_rdf[:, 2]))
    )
    fov = np.maximum(angles_vertical.max(), angles_horizontal.max())
    # Projecting the corners into a virtual camera with resolution 1. You'd need to multiply the resulting values with
    # the resolution to get the new 2d box coordinates.
    # If we want to keep the original resolution: new_box_width = new_resolution * reprojected_width = old_box_width
    resolutionless_intrinsic = np.array(
        [
            [1.0 / (2 * np.tan(np.deg2rad(fov) / 2)), 0, 0],
            [0, 1.0 / (2 * np.tan(np.deg2rad(fov) / 2)), 0],
            [0, 0, 1],
        ]
    )
    reprojected_corners = project_points_3d_to_2d(
        k_matrix=resolutionless_intrinsic,
        camera_model=CAMERA_MODEL_OPENCV_PINHOLE,
        distortion_lookup=None,
        distortion_parameters=None,
        points_3d=corners_3d_in_virtual_cam_rdf,
    )
    reprojected_dimensions = reprojected_corners.ptp(axis=0)[:2]

    max_dimension = reprojected_dimensions.max()
    if max_dimension > 1.0:
        # Don't want to downscale anything
        reprojected_dimensions /= max_dimension
    scale_factors = np.array([clipped_width, clipped_height]) / reprojected_dimensions
    resolution = np.ceil(scale_factors.max()).astype(np.int64)
    return fov, resolution


def _get_bounding_box_from_mask(input_mask: np.ndarray) -> Tuple[float, float, float, float]:
    y, x = np.where(input_mask[:, :, 0])
    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + 0.5 * width
    center_y = min_y + 0.5 * height
    return center_x, center_y, height, width
