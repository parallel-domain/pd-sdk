from typing import List, Optional, Union

import numpy as np
from pyquaternion import Quaternion

from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem


class Transformation:
    """Multi-purpose 6DoF object.

    When printed, the rotation is given as euler angles `[yaw, pitch, roll]` following the z-y'-x'' convention.

    Args:
        quaternion: Quaternion instance for rotation. Default: Unit quaternion without rotation.
        translation: List-like translation information in order `(x,y,z)`. Default: `[0,0,0]`

    Example:
        `Transformation` instances can be easily matrix-multiplied with other `Transformation` instances or any
        `np.ndarray` of shape (4,n).

        ::

            lidar_frame = ...  # get any `SensorFrame` from a LiDAR sensor
            points_vehicle_frame = (lidar_frame.extrinsic @ lidar_frame.xyz_one.T).T
            points_world_frame = (lidar_frame.pose @ lidar_frame.xyz_one.T).T

            boxes_3d = lidar_frame.get_annotations(AnnotationTypes.BoundingBoxes3D)

            for b in boxes_3d.boxes:
                box_pose_world_frame = lidar_frame.pose @ b.pose
    """

    def __init__(self, quaternion: Quaternion = None, translation: Union[np.array, List] = None):
        self._Rq = quaternion if quaternion is not None else Quaternion(w=1, x=0, y=0, z=0)
        self._t = np.asarray(translation).reshape(3) if translation is not None else np.array([0, 0, 0])

    def __repr__(self):
        rep = f"R: {list(self.quaternion.yaw_pitch_roll)}, t: {self.translation}"
        return rep

    def __matmul__(self, other) -> Union["Transformation", np.ndarray]:
        convert_to_transform = True
        if isinstance(other, Transformation):
            transform = self.transformation_matrix @ other.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = self.transformation_matrix @ other
            convert_to_transform = other.shape == (4, 4)
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")
        if convert_to_transform:
            return Transformation.from_transformation_matrix(mat=transform)
        return transform

    def __rmatmul__(self, other) -> Union["Transformation", np.ndarray]:
        convert_to_transform = True
        if isinstance(other, Transformation):
            transform = other.transformation_matrix @ self.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = other @ self.transformation_matrix
            convert_to_transform = other.shape == (4, 4)
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")

        if convert_to_transform:
            return Transformation.from_transformation_matrix(mat=transform)
        return transform

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Returns the homogeneous transformation matrix in shape (4,4).

        ::

            /                  \\
            |R_11 R_12 R_13 t_x|
            |R_21 R_22 R_23 t_y|
            |R_31 R_32 R_33 t_z|
            |0    0    0    1  |
            \\                  /
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def rotation(self) -> np.ndarray:
        """Returns the rotation matrix in shape (3,3).

        ::

            /              \\
            |R_11 R_12 R_13|
            |R_21 R_22 R_23|
            |R_31 R_32 R_33|
            \\              /
        """
        return self._Rq.rotation_matrix

    @property
    def quaternion(self) -> Quaternion:
        """Returns the rotation as a :obj:`.pyquaternion.quaternion.Quaternion` instance.

        Full documentation can be found in
        `pyquaternion API Documentation <http://kieranwynn.github.io/pyquaternion/>`_.

        To get the quaternion coefficients, either call `.elements`, iterate over the object itself or use
        the dedicated named properties. The element order (until explicitly stated otherwise) should always be assumed
        as `(w,x,y,z)` for function `w + xi+ yj + zk`

        ::

            from paralleldomain.model.transformation import Transformation

            tf = Transformation.from_euler_angles(yaw=90, pitch=0, roll=0, is_degrees=True)

            assert(tf.quaternion.elements[0] == tf.quaternion[0] == tf.quaternion.w)
            assert(tf.quaternion.elements[1] == tf.quaternion[1] == tf.quaternion.x)
            assert(tf.quaternion.elements[2] == tf.quaternion[2] == tf.quaternion.y)
            assert(tf.quaternion.elements[3] == tf.quaternion[3] == tf.quaternion.z)

        Please note that when using :obj:`.scipy.spatial.transform.Rotation`, `scipy` assumes the order as `(x,y,w,z)`.

        ::

            from paralleldomain.model.transformation import Transformation
            from scipy.spatial.transform import Rotation
            import numpy as np

            tf = Transformation.from_euler_angles(yaw=90, pitch=0, roll=0, is_degrees=True)
            tf_scipy = Rotation.from_quat([
                tf.quaternion.x,
                tf.quaternion.y,
                tf.quaternion.z,
                tf.quaternion.w
            ])

            # Check that rotation quaternion is equal within tolerance
            np.allclose(tf.rotation == tf_scipy.as_matrix())  # returns True

        """
        return self._Rq

    @property
    def translation(self) -> np.array:
        """Returns the translation vector `(x,y,z)` in shape (3,)."""
        return self._t

    @property
    def inverse(self) -> "Transformation":
        """Returns the inverse transformation as a new :obj:`Transformation` object."""
        q_inv = self.quaternion.inverse
        t_inv = -1 * q_inv.rotate(self.translation)
        T_inv = Transformation(q_inv, t_inv)
        return T_inv

    @classmethod
    def from_transformation_matrix(cls, mat: np.ndarray) -> "Transformation":
        """Creates a Transformation object from an homogeneous transformation matrix of shape (4,4)

        Args:
            mat: Transformation matrix as described in :attr:`~.Transformation.transformation_matrix`

        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        quat = Quaternion(matrix=mat)
        translation = mat[:3, 3]
        return cls(quaternion=quat, translation=translation)

    @classmethod
    def from_euler_angles(
        cls,
        roll: Optional[float] = 0.0,
        pitch: Optional[float] = 0.0,
        yaw: Optional[float] = 0.0,
        translation: Optional[np.ndarray] = None,
        degrees: bool = False,
        order: str = "xyz",
        coordinate_system: Optional[Union[str, CoordinateSystem]] = None,
    ) -> "Transformation":
        """Creates a transformation object from euler angles and optionally translation (default: (0,0,0))

        Args:
            roll: Rotation angle around x-axis. Default: `0.0`
            pitch: Rotation angle around y-axis. Default: `0.0`
            yaw: Rotation angle around z-axis. Default: `0.0`
            translation: Translation vector in order `(x,y,z)`. Default: `[0,0,0]`
            degrees: Defines if euler angles are provided in degrees instead of radians. Default: `False`
            order: Set intrinsic rotation order. Default: `xyz`
            coordinate_system: Set custom coordinate system. Default: `FLU`

        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])

        if coordinate_system is None:
            coordinate_system = INTERNAL_COORDINATE_SYSTEM
        elif isinstance(coordinate_system, str):
            coordinate_system = CoordinateSystem(axis_directions=coordinate_system)

        quat = coordinate_system.quaternion_from_rpy(roll=roll, pitch=pitch, yaw=yaw, degrees=degrees, order=order)
        return cls(quaternion=quat, translation=translation)
