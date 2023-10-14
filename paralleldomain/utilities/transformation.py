from typing import List, Optional, TypeVar, Union, TYPE_CHECKING

import numpy as np
from pyquaternion import Quaternion
from transforms3d.euler import euler2mat, mat2euler

if TYPE_CHECKING:
    from paralleldomain.utilities.coordinate_system import CoordinateSystem

T = TypeVar("T")


class Transformation:
    """Multi-purpose 6DoF object.

    When printed, the rotation is given as euler angles in degrees following intrinsic rotation order XYZ,
    rounded to 2 decimal places.

    Args:
        quaternion: Quaternion instance or elements for rotation. Elements are expected in order `(w,x,y,z)`.
            Default: Unit quaternion without rotation.
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

    def __init__(
        self, quaternion: Union[Quaternion, List[float], np.ndarray] = None, translation: Union[np.ndarray, List] = None
    ):
        if quaternion is None:
            self._Rq = Quaternion(w=1, x=0, y=0, z=0)
        else:
            if isinstance(quaternion, Quaternion):
                self._Rq = quaternion
            elif isinstance(quaternion, (List, np.ndarray)):
                self._Rq = Quaternion(**dict(zip(["w", "x", "y", "z"], np.asarray(quaternion).reshape(4))))
            else:
                raise TypeError("quaternion only accepts (pyquaternion.Quaternion, List[float], np.ndarray.")

        self._t = np.asarray(translation).reshape(3) if translation is not None else np.array([0, 0, 0])

    def __repr__(self):
        rep = (
            f"R: {list(map(round,self.as_euler_angles(order='XYZ', degrees=True),3*[2]))},"
            f"t: {list(map(round,self.translation,3*[2]))}"
        )
        return rep

    def __matmul__(self, other: T) -> T:
        if isinstance(other, Transformation):
            # (s @ o) @ x = s.r @ (o.r @ x + o.t) + s.t = (s.r @ o.r) @ x + s.r @ o.t + s.t
            rotation = self.quaternion * other.quaternion
            translation = self.quaternion.rotate(other.translation) + self.translation
            return Transformation(quaternion=rotation, translation=translation)
        elif isinstance(other, np.ndarray):
            if (len(other.shape) == 1 and other.shape[0] == 3) or (len(other.shape) == 2 and other.shape[1] == 3):
                if len(other.shape) == 2 and other.shape[1] == 3:
                    other = np.concatenate([other, np.ones_like(other[:, :1])], -1)
                    transform = self.transformation_matrix @ other.T
                    return transform[:3, :].T
                else:
                    other = np.expand_dims(np.array([*other, 1.0]), 1)
                    transform = self.transformation_matrix @ other
                    return transform[:3, 0]

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
    def from_transformation_matrix(cls, mat: np.ndarray, approximate_orthogonal: bool = False) -> "Transformation":
        """Creates a Transformation object from an homogeneous transformation matrix of shape (4,4)

        Args:
            mat: Transformation matrix as described in :attr:`~.Transformation.transformation_matrix`
            approximate_orthogonal: When set to `True`, non-orthogonal matrices will be approximate to their closest
                orthogonal representation. Default: `False`.

        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        if approximate_orthogonal:
            # Implemented after `scipy.spatial.transform.rotation.from_matrix()`
            decision = np.zeros(shape=(4,))

            decision[0] = mat[0, 0]
            decision[1] = mat[1, 1]
            decision[2] = mat[2, 2]
            decision[3] = mat[0, 0] + mat[1, 1] + mat[2, 2]
            choice = np.argmax(decision)

            quat_coefficients = np.zeros(shape=(4,))

            if choice != 3:
                i = choice
                j = (i + 1) % 3
                k = (j + 1) % 3

                quat_coefficients[i] = 1 - decision[3] + 2 * mat[i, i]
                quat_coefficients[j] = mat[j, i] + mat[i, j]
                quat_coefficients[k] = mat[k, i] + mat[i, k]
                quat_coefficients[3] = mat[k, j] - mat[j, k]
            else:
                quat_coefficients[0] = mat[2, 1] - mat[1, 2]
                quat_coefficients[1] = mat[0, 2] - mat[2, 0]
                quat_coefficients[2] = mat[1, 0] - mat[0, 1]
                quat_coefficients[3] = 1 + decision[3]

            quat_coefficients = quat_coefficients / np.linalg.norm(quat_coefficients)

            quat = Quaternion(
                x=quat_coefficients[0], y=quat_coefficients[1], z=quat_coefficients[2], w=quat_coefficients[3]
            )
        else:
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

    def as_yaw_pitch_roll(self, coordinate_system: "CoordinateSystem", degrees: bool = False) -> np.ndarray:
        """Returns the rotation of a `Transformation` object as yaw pitch roll euler angles.
        Please note that the direction of a positive angle is dependent on the given coordinate_system. For example a
        positive pitch only rotates upwards if the right axis is part of the coordinate system directions.

        Args:
            coordinate_system: CoordinateSystem the Transformation is in. Determines the correct euler axis order.
            degrees: Defines if euler angles should be returned in degrees instead of radians. Default: `False`
        Returns:
            Array contain [yaw angle, pitch angle, roll angle]
        """
        axis_order = coordinate_system.get_yaw_pitch_roll_order_string()
        return self.as_euler_angles(order=axis_order, degrees=degrees)[::-1]

    def apply_to(self, points_3d: np.ndarray) -> np.ndarray:
        return apply_transform_3d(tf=self, points_3d=points_3d)

    @classmethod
    def interpolate(cls, tf0: "Transformation", tf1: "Transformation", factor: float = 0.5) -> "Transformation":
        """Interpolates the translation and rotation between two `Transformation` objects.

        For translation, linear interpolation is used:
            `tf0.translation + factor * (tf1.translation - tf0.translation)`. For rotation, spherical linear
            interpolation of rotation quaternions is used:
            `tf0.quaternion * (conjugate(tf0.quaternion) * tf1.quaternion)**factor`

        Args:
            tf0: First `Transformation` object used as interpolation start
            tf1: Second `Transformation` object used as interpolation end
            factor: Interpolation factor within `[0.0, 1.0]`. If `0.0`, the return value is equal to `tf0`;
                 if `1.0`, the return value is equal to `tf1`. Values smaller than `0.0` or greater than `1.0` can be
                 used if extrapolation is desired. Default: `0.5`
        Returns:
            A new `Transformation` object that is at the interpolated factor between `tf0` and `tf1`.
        """

        def lerp(p0: np.ndarray, p1: np.ndarray, factor: float) -> np.ndarray:
            factors = np.asarray([1 - factor, factor])
            points = np.vstack([p0, p1])
            return factors @ points

        def slerp(p: Quaternion, q: Quaternion, factor: float) -> Quaternion:
            return p * (p.conjugate * q) ** factor

        return cls(
            translation=lerp(p0=tf0.translation, p1=tf1.translation, factor=factor),
            quaternion=slerp(p=tf0.quaternion, q=tf1.quaternion, factor=factor),
        )

    @classmethod
    def from_euler_angles(
        cls,
        angles: Union[np.ndarray, List[float]],
        order: str,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
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
        elif isinstance(translation, list):
            translation = np.array(translation)
        if degrees:
            angles = np.deg2rad(angles)

        assert len(order) == 3
        if order.isupper():
            tf3d_order = f"r{order.lower()}"
        else:
            tf3d_order = f"s{order.lower()}"

        mat = euler2mat(ai=angles[0], aj=angles[1], ak=angles[2], axes=tf3d_order)
        quat = Quaternion(matrix=mat)

        return cls(quaternion=quat, translation=translation)

    @classmethod
    def from_yaw_pitch_roll(
        cls,
        coordinate_system: "CoordinateSystem",
        yaw: float = 0,
        pitch: float = 0,
        roll: float = 0,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
        degrees: bool = False,
    ) -> "Transformation":
        """
        Creates a transformation object from yaw pitch roll angles and optionally translation (default: (0,0,0)).
        The returned transformation first rotates by roll around the front/back axis, then by pitch around the
        left/right axis and finally by yaw around the up/down axis (from an extrinsic perspective).
        Please note that the direction of a positive angle is dependent on the given coordinate_system. For example a
        positive pitch only rotates upwards if the right axis is part of the coordinate system directions.

        Args:
            coordinate_system: determines the correct euler rotation axis order.
            yaw: rotation around up/down
            pitch: rotation around left/right
            roll: rotation round front/back
            translation: Translation vector in order `(x,y,z)`. Default: `[0,0,0]`
            degrees: Defines if euler angles are provided in degrees instead of radians. Default: `False`

        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        order_string = coordinate_system.get_yaw_pitch_roll_order_string()
        return cls.from_euler_angles(
            angles=[roll, pitch, yaw], order=order_string, translation=translation, degrees=degrees
        )

    @classmethod
    def from_axis_angle(
        cls,
        axis: Union[np.ndarray, List[float]],
        angle: float,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
        degrees: bool = False,
    ) -> "Transformation":
        """Creates a transformation object from axis and angle, and optionally translation (default: (0,0,0))

        Args:
            axis: A vector that represents the axis of rotation. Will be normalized, if not already.
            angle: The angle of rotation in radians. If `degrees` is `True`, this value is expected to be in degrees.
            translation: Translation vector in order `(x,y,z)`. Default: `[0,0,0]`
            degrees: Defines if euler angles are provided in degrees instead of radians. Default: `False`
        Returns:
            Instance of :obj:`Transformation` with provided parameters.
        """
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])
        elif isinstance(translation, list):
            translation = np.array(translation)
        if degrees:
            angle = np.deg2rad(angle)

        axis = axis / np.linalg.norm(axis)
        quat = Quaternion(axis=axis, angle=angle)

        return cls(quaternion=quat, translation=translation)

    @staticmethod
    def look_at(
        target: Union[np.ndarray, List[float]],
        coordinate_system: str,
        position: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> "Transformation":
        """
        Calculates the pose transformation of being located at the given positions and looking at the given target.
        Args:
            target: The position to look at
            coordinate_system: The coordinate system the result, target and position are in
            position: the position to look from

        Returns:
            Instance of :obj:`Transformation` looking from position to target
        """
        from paralleldomain.utilities.coordinate_system import CoordinateSystem

        if position is None:
            position = [0, 0, 0]
        position = np.asarray(position)
        # Calculate it in flu, convert it to something else later
        to_flu_transformation = CoordinateSystem.get_base_change_from_to(
            from_axis_directions=coordinate_system, to_axis_directions="FLU"
        )
        front_direction = np.asarray(target) - position
        front_direction = (to_flu_transformation @ front_direction[np.newaxis, :])[0]
        front_direction /= np.linalg.norm(front_direction)

        left_direction = np.cross(front_direction, [0, 0, -1])
        left_direction /= np.linalg.norm(left_direction)

        up_direction = np.cross(front_direction, left_direction)
        up_direction /= np.linalg.norm(up_direction)

        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = np.array([front_direction, left_direction, up_direction]).T
        transformation_matrix[:3, 3] = (to_flu_transformation @ position[np.newaxis, :])[0]

        transformation_in_flu = Transformation.from_transformation_matrix(mat=transformation_matrix)
        return CoordinateSystem.change_transformation_coordinate_system(
            transformation=transformation_in_flu, transformation_system="FLU", target_system=coordinate_system
        )


def apply_transform_3d(tf: Transformation, points_3d: np.ndarray) -> np.ndarray:
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(
            f"""Expected np.ndarray of shape (N X 3) for `points_3d`, where N is
                   number of points. Received {points_3d.shape}."""
        )

    return (tf @ (np.hstack([points_3d, np.ones(shape=(len(points_3d), 1))])).T).T[:, :3]
