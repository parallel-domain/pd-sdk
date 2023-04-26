import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from paralleldomain.decoding.waymo_open_dataset.common import WAYMO_LIDAR_NAME_TO_INDEX, WAYMO_USE_ALL_LIDAR_NAME

# from waymo_open_dataset import dataset_pb2
from paralleldomain.decoding.waymo_open_dataset.protos import dataset_pb2

RangeImages = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixFloat]]
CameraProjections = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixInt32]]
SegmentationLabels = Dict["dataset_pb2.LaserName.Name", List[dataset_pb2.MatrixInt32]]
ParsedFrame = Tuple[RangeImages, CameraProjections, SegmentationLabels, Optional[dataset_pb2.MatrixFloat]]

"""
These functions were brought in from waymo_open_datset frame_utils.py, range_utils.py, and transform_utils.py
The only adjustment was the removal of the dependency on tensorflow, replaced with numpy functions.
"""


def parse_single_lidar_scanner(
    laser: dataset_pb2.Laser, range_images: Dict, seg_labels: Dict, range_image_top_pose: dataset_pb2.MatrixFloat
) -> Tuple[Dict, Dict, dataset_pb2.MatrixFloat]:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
        range_image_str_tensor = zlib.decompress(laser.ri_return1.range_image_compressed)
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(bytearray(range_image_str_tensor))
        range_images[laser.name] = [ri]

        if laser.name == dataset_pb2.LaserName.TOP:
            range_image_top_pose_str_tensor = zlib.decompress(laser.ri_return1.range_image_pose_compressed)
            range_image_top_pose = dataset_pb2.MatrixFloat()
            range_image_top_pose.ParseFromString(bytearray(range_image_top_pose_str_tensor))

        if len(laser.ri_return1.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
            seg_label_str_tensor = zlib.decompress(laser.ri_return1.segmentation_label_compressed)
            seg_label = dataset_pb2.MatrixInt32()
            seg_label.ParseFromString(bytearray(seg_label_str_tensor))
            seg_labels[laser.name] = [seg_label]
    if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
        range_image_str_tensor = zlib.decompress(laser.ri_return2.range_image_compressed)
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(bytearray(range_image_str_tensor))
        range_images[laser.name].append(ri)

        if len(laser.ri_return2.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
            seg_label_str_tensor = zlib.decompress(laser.ri_return2.segmentation_label_compressed)
            seg_label = dataset_pb2.MatrixInt32()
            seg_label.ParseFromString(bytearray(seg_label_str_tensor))
            seg_labels[laser.name].append(seg_label)
    return range_images, seg_labels, range_image_top_pose


def parse_range_image_and_camera_projection(
    record: dataset_pb2.Frame, sensor_name: str = WAYMO_USE_ALL_LIDAR_NAME
) -> Tuple[Dict, Dict, dataset_pb2.MatrixFloat]:
    """Parse range images and camera projections given a frame.

    Args:
      record: open dataset frame proto

    Returns:
      range_images: A dict of {laser_name,
        [range_image_first_return, range_image_second_return]}.
      seg_labels: segmentation labels, a dict of {laser_name,
        [seg_label_first_return, seg_label_second_return]}
      range_image_top_pose: range image pixel pose for top lidar.
    """
    range_images = {}
    seg_labels = {}
    range_image_top_pose = None
    if sensor_name == WAYMO_USE_ALL_LIDAR_NAME:
        # Loop through all 5 laser scanners
        for laser in record.lasers:
            range_images, seg_labels, range_image_top_pose = parse_single_lidar_scanner(
                laser=laser, range_images=range_images, seg_labels=seg_labels, range_image_top_pose=range_image_top_pose
            )
    else:
        raise KeyError(
            "For Waymo Open Dataset, all LiDAR sensors are combined. Single sensors are not currently supported."
        )
    return range_images, seg_labels, range_image_top_pose


def convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, ri_index=0):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      dict of {laser_name, (H, W, D)} range images in Cartesian coordinates.
        5 fields are [x, y, z, intensity, elongation]
    """
    cartesian_range_images = {}
    frame_pose = np.reshape(np.array(frame.pose.transform), [4, 4])

    # [H, W, 6]
    range_image_top_pose_tensor = np.asarray(range_image_top_pose.data).reshape(range_image_top_pose.shape.dims)

    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[..., 2]
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = compute_inclination(
                np.array([c.beam_inclination_min, c.beam_inclination_max]), height=range_image.shape.dims[0]
            )
        else:
            beam_inclinations = np.array(c.beam_inclinations)

        beam_inclinations = np.flip(beam_inclinations, axis=-1)
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = np.asarray(range_image.data).reshape(range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = np.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = np.expand_dims(frame_pose, axis=0)
        range_image_cartesian = extract_point_cloud_from_range_image(
            np.expand_dims(range_image_tensor[..., 0], axis=0),
            np.expand_dims(extrinsic, axis=0),
            np.expand_dims(beam_inclinations, axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_cartesian = np.squeeze(range_image_cartesian, axis=0)

        # Note: not currently keeping range because it's repetitive with x,y,z., but it is available here
        range_image_cartesian = np.concatenate([range_image_cartesian, range_image_tensor[..., 1:3]], axis=-1)

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images


def convert_range_image_to_point_cloud(frame, range_images, range_image_top_pose, ri_index=0):
    """Convert range images to point cloud.
    Note: camera projection points were removed from this function but were present in Waymo Open Dataset version.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      points: {[N, 5]} list of 3d lidar points of length 5 (number of lidars).
        5 fields are [x, y, z, intensity, elongation]
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []

    cartesian_range_images = convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, ri_index)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = np.asarray(range_image.data).reshape(range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = range_image_cartesian[range_image_mask]

        points.append(points_tensor)

    return points


def get_rotation_matrix(roll, pitch, yaw, name=None):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
      roll : x-rotation in radians.
      pitch: y-rotation in radians. The shape must be the same as roll.
      yaw: z-rotation in radians. The shape must be the same as roll.
      name: the op name.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack(
        [
            np.stack([ones, zeros, zeros], axis=-1),
            np.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
            np.stack([zeros, sin_roll, cos_roll], axis=-1),
        ],
        axis=-2,
    )
    r_pitch = np.stack(
        [
            np.stack([cos_pitch, zeros, sin_pitch], axis=-1),
            np.stack([zeros, ones, zeros], axis=-1),
            np.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
        ],
        axis=-2,
    )
    r_yaw = np.stack(
        [
            np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
            np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
            np.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    return np.matmul(r_yaw, np.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Args:
      rotation: [..., N, N] rotation tensor.
      translation: [..., N] translation tensor. This must have the same type as
        rotation.

    Returns:
      transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    # [..., N, 1]
    translation_n_1 = translation[..., np.newaxis]
    # [..., N, N+1]
    transform = np.concatenate([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = np.zeros_like(translation)
    # [..., N+1]
    last_row = np.concatenate([last_row, np.ones_like(last_row[..., 0:1])], axis=-1)
    # [..., N+1, N+1]
    transform = np.concatenate([transform, last_row[..., np.newaxis, :]], axis=-2)
    return transform


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=np.float32, scope=None):
    """Computes range image polar coordinates.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = range_image.shape
    range_image_dtype = range_image.dtype
    range_image = range_image.astype(dtype)
    extrinsic = extrinsic.astype(dtype)
    inclination = inclination.astype(dtype)

    # [B].
    az_correction = np.arctan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
    # [W].
    ratios = (np.arange(width, 0, -1).astype(dtype) - 0.5) / np.array(width).astype(dtype)
    # [B, W].
    azimuth = (ratios * 2.0 - 1.0) * np.pi - np.expand_dims(az_correction, -1)

    # [B, H, W]
    azimuth_tile = np.tile(azimuth[:, np.newaxis, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = np.tile(inclination[:, :, np.newaxis], [1, 1, width])
    range_image_polar = np.stack([azimuth_tile, inclination_tile, range_image], axis=-1)
    return range_image_polar.astype(range_image_dtype)


def compute_range_image_cartesian(
    range_image_polar, extrinsic, pixel_pose=None, frame_pose=None, dtype=np.float32, scope=None
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = range_image_polar.astype(dtype)
    extrinsic = extrinsic.astype(dtype)
    if pixel_pose is not None:
        pixel_pose = pixel_pose.astype(dtype)
    if frame_pose is not None:
        frame_pose = frame_pose.astype(dtype)

    azimuth, inclination, range_image_range = np.moveaxis(range_image_polar, -1, 0)

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = np.stack([x, y, z], -1)
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = np.expand_dims(np.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = np.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
    if pixel_pose is not None:
        # To global frame.
        # [B, H, W, 3, 3]
        pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
        # [B, H, W, 3]
        pixel_pose_translation = pixel_pose[..., 0:3, 3]
        # [B, H, W, 3]
        range_image_points = (
            np.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
        )
        if frame_pose is None:
            raise ValueError("frame_pose must be set when pixel_pose is set.")
        # To vehicle frame corresponding to the given frame_pose
        # [B, 4, 4]
        world_to_vehicle = np.linalg.inv(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
        world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
        # [B, H, W, 3]
        range_image_points = (
            np.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
            + world_to_vehicle_translation[:, np.newaxis, np.newaxis, :]
        )

    range_image_points = range_image_points.astype(range_image_polar_dtype)
    return range_image_points


def extract_point_cloud_from_range_image(
    range_image, extrinsic, inclination, pixel_pose=None, frame_pose=None, dtype=np.float32, scope=None
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle
      frame.
    """
    range_image_polar = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar, extrinsic, pixel_pose=pixel_pose, frame_pose=frame_pose, dtype=dtype
    )
    return range_image_cartesian


def compute_inclination(inclination_range, height, scope=None):
    """Computes uniform inclination range based the given range and height.

    Args:
      inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
        inclination].
      height: an integer indicates height of the range image.
      scope: the name scope.

    Returns:
      inclination: [..., height] tensor. Inclinations computed.
    """
    height = np.array(height)
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (0.5 + np.arange(0, height).astype(inclination_range.dtype)) / height.astype(
        inclination_range.dtype
    ) * np.expand_dims(diff, axis=-1) + inclination_range[..., 0:1]
    return inclination
