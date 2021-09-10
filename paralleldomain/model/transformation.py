from typing import List, Optional, TypeVar, Union

import numpy as np
from pyquaternion import Quaternion
from transforms3d.euler import euler2mat, mat2euler

T = TypeVar("T")


class Transformation:
    """Multi-purpose 6DoF object.

    When printed, the rotation is given as euler angles in degrees following intrinsic rotation order XYZ,
    rounded to 2 decimal places.

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

    def __init__(self, quaternion: Quaternion = None, translation: Union[np.ndarray, List] = None):
        self._Rq = quaternion if quaternion is not None else Quaternion(w=1, x=0, y=0, z=0)
        self._t = np.asarray(translation).reshape(3) if translation is not None else np.array([0, 0, 0])

    def __repr__(self):
        rep = f"R: {list(map(round,self.as_euler_angles(order='XYZ', degrees=True),3*[2]))}, t: {self.translation}"
        return rep

    def __matmul__(self, other: T) -> T:
        if isinstance(other, Transformation):
            transform = self.transformation_matrix @ other.transformation_matrix
            return Transformation.from_transformation_matrix(mat=transform)
        elif isinstance(other, np.ndarray):
            transform = self.transformation_matrix @ other
            return transform
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4xn numpy array!")

    def __rmatmul__(self, other: T) -> T:  # Union["Transformation", np.ndarray]:
        if isinstance(other, Transformation):
            transform = other.transformation_matrix @ self.transformation_matrix
            return Transformation.from_transformation_matrix(mat=transform)
        elif isinstance(other, np.ndarray):
            transform = other @ self.transformation_matrix
            return transform
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or nx4 numpy array!")

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
    def translation(self) -> np.ndarray:
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

    def as_euler_angles(self, order: str, degrees: bool = False) -> np.ndarray:
        """Returns the rotation of a `Transformation` object as euler angles.

        Args:
            order: Defines the axes rotation order. Use lower case for extrinsic rotation, upper case for intrinsic
               rotation. Ex: `xyz`, `ZYX`, `xzx`.
            degrees: Defines if euler angles should be returned in degrees instead of radians. Default: `False`
        Returns:
            Ordered array of euler angles with length 3.
        """
        assert len(order) == 3
        if order.isupper():
            tf3d_order = f"r{order.lower()}"
        else:
            tf3d_order = f"s{order.lower()}"

        angles = np.asarray(mat2euler(mat=self.rotation, axes=tf3d_order))

        if degrees:
            return np.rad2deg(angles)
        else:
            return angles

    @classmethod
    def from_euler_angles(
        cls,
        angles: Union[np.ndarray, List[float]],
        order: str,
        translation: Optional[np.ndarray] = None,
        degrees: bool = False,
    ) -> "Transformation":
        """Creates a transformation object from euler angles and optionally translation (default: (0,0,0))

        Args:
            angles: Ordered euler angles array with length 3
            translation: Translation vector in order `(x,y,z)`. Default: `[0,0,0]`
            order: Defines the axes rotation order. Use lower case for extrinsic rotation, upper case for intrinsic
               rotation. Ex: `xyz`, `ZYX`, `xzx`.
            degrees: Defines if euler angles are provided in degrees instead of radians. Default: `False`
        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])
        if degrees:
            angles = np.deg2rad(angles)

        assert len(order) == 3
        if order.isupper():
            tf3d_order = f"r{order.lower()}"
        else:
            tf3d_order = f"s{order.lower()}"

        mat = euler2mat(ai=angles[0], aj=angles[1], ak=angles[2], axes=tf3d_order)
        quat = Quaternion(matrix=mat)

        return Transformation(quaternion=quat, translation=translation)
