from typing import Union

import numpy as np

from paralleldomain.model.annotation import AnnotationPose, BoundingBox2D, BoundingBox3D


def merge_boxes(
    target_box: Union[BoundingBox2D, BoundingBox3D], source_box: Union[BoundingBox2D, BoundingBox3D]
) -> Union[BoundingBox2D, BoundingBox3D]:
    """
    Takes two boxes as input and merges both into a new box.
    The resulting box has the exact same properties as `target_box`,
    but with extended `source_box` dimensions merged into it.
    """
    if type(target_box) == BoundingBox3D and type(source_box) == BoundingBox2D:
        return merge_boxes_3d(target_box=target_box, source_box=source_box)
    elif type(target_box) == BoundingBox2D and type(source_box) == BoundingBox2D:
        return merge_boxes_2d(target_box=target_box, source_box=source_box)
    else:
        TypeError(f"Types {type(target_box)} for target_box and {type(source_box)} for source_box must be the same.")


def merge_boxes_2d(target_box: BoundingBox2D, source_box: BoundingBox2D) -> BoundingBox2D:
    """
    Takes two 2D boxes as input and merges both into a new box.
    The resulting box has the exact same properties as `target_box`,
    but with extended `source_box` dimensions merged into it.
    """
    x_coords = []
    y_coords = []
    for b in [target_box, source_box]:
        x_coords.append(b.x)
        x_coords.append(b.x + b.width)
        y_coords.append(b.y)
        y_coords.append(b.y + b.height)

    x_ul_new = min(x_coords)
    x_width_new = max(x_coords) - x_ul_new
    y_ul_new = min(y_coords)
    y_height_new = max(y_coords) - y_ul_new

    result_box = BoundingBox2D(
        x=x_ul_new,
        y=y_ul_new,
        width=x_width_new,
        height=y_height_new,
        class_id=target_box.class_id,
        instance_id=target_box.instance_id,
        attributes=target_box.attributes,
    )

    return result_box


def merge_boxes_3d(target_box: BoundingBox3D, source_box: BoundingBox3D) -> BoundingBox3D:
    """
    Takes two 3D boxes as input and merges both into a new box.
    The resulting box has the exact same properties as `target_box`,
    but with extended `source_box` dimensions merged into it.
    """

    source_faces = np.array(
        [
            [source_box.length / 2, 0.0, 0.0, 1.0],
            [-1 * source_box.length / 2, 0.0, 0.0, 1.0],
            [0.0, source_box.width / 2, 0.0, 1.0],
            [0.0, -1.0 * source_box.width / 2, 0.0, 1.0],
            [0.0, 0.0, source_box.height / 2, 1.0],
            [0.0, 0.0, -1 * source_box.height / 2, 1.0],
        ]
    )
    target_faces = np.array(
        [
            [target_box.length / 2, 0.0, 0.0, 1.0],
            [-1 * target_box.length / 2, 0.0, 0.0, 1.0],
            [0.0, target_box.width / 2, 0.0, 1.0],
            [0.0, -1.0 * target_box.width / 2, 0.0, 1.0],
            [0.0, 0.0, target_box.height / 2, 1.0],
            [0.0, 0.0, -1 * target_box.height / 2, 1.0],
        ]
    )
    sensor_frame_faces = source_box.pose @ source_faces.transpose()
    bike_frame_faces = (target_box.pose.inverse @ sensor_frame_faces).transpose()
    max_faces = np.where(np.abs(target_faces) > np.abs(bike_frame_faces), target_faces, bike_frame_faces)
    length = max_faces[0, 0] - max_faces[1, 0]
    width = max_faces[2, 1] - max_faces[3, 1]
    height = max_faces[4, 2] - max_faces[5, 2]
    center = np.array(
        [max_faces[1, 0] + 0.5 * length, max_faces[3, 1] + 0.5 * width, max_faces[5, 2] + 0.5 * height, 1.0]
    )
    translation = target_box.pose @ center
    fused_pose = AnnotationPose(quaternion=target_box.pose.quaternion, translation=translation[:3])
    attributes = target_box.attributes

    result_box = BoundingBox3D(
        pose=fused_pose,
        length=length,
        width=width,
        height=height,
        class_id=target_box.class_id,
        instance_id=target_box.instance_id,
        num_points=(target_box.num_points + source_box.num_points),
        attributes=attributes,
    )

    return result_box
