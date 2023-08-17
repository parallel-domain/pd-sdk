import logging
import math
from typing import Dict

import numpy as np
from pyquaternion import Quaternion

from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


class CoordinateSystem:
    """A class to represent a 3D coordinate system and perform transformation between different coordinate systems.

    Attributes:
        axis_directions (str): A string representing the three orthogonal directions of the coordinate system.
        _base_matrix (np.ndarray): A 4x4 numpy array representing the matrix used to transform coordinates from
            the base coordinate system to the current coordinate system.
        is_right_handed (bool): A boolean indicating whether the coordinate system is right-handed or not.

    Methods:
        get_base_change_from_to(from_axis_directions: str, to_axis_directions: str): transforms a vector from
            the current coordinate system to another
        print_convention(): Prints the convention used to define the front, left and up axis of the coord
        quaternion_from_rpy(self, roll: float, pitch: float, yaw: float, degrees: bool = False, order: str = "xyz"):
            Convert the given roll, pitch and yaw angles into a quaternion representation

    """

    _flip_map = {"B": "F", "F": "B", "L": "R", "R": "L", "U": "D", "D": "U"}
    _axis_char_map: Dict[str, np.ndarray] = dict(
        **{character: axis for character, axis in zip("FLU", np.identity(3))},
        **{character: axis for character, axis in zip("BRD", -np.identity(3))},
    )

    def __init__(self, axis_directions: str):
        """A class to represent a 3D coordinate system and perform transformation between
        different coordinate systems.

        Args:
            axis_directions (str): A string representing the three orthogonal directions of the coordinate system.
        """
        self.axis_directions = axis_directions
        self._base_matrix = self._create_base_matrix(axis_directions=axis_directions)
        self.is_right_handed = np.linalg.det(self._base_matrix) == 1

    @staticmethod
    def _create_base_matrix(axis_directions: str) -> np.ndarray:
        """Create a base matrix for the coordinate system based on the given axis direction

        Args:
            axis_directions(str): A string representing the three orthogonal directions of the coordinate system

        Returns:
            np.ndarray: A 4x4 numpy array representing the matrix used to transform coordinates from the base
                coordinate system to the current coordinate system.

        Raises:
            ValueError: If the axis string does not have exactly 3 orthogonal values from {RUBLDF}, or if the
                resulting matrix has determinant 0 (invalid coordinate system).

        """
        if len(axis_directions) != 3:
            raise ValueError("The axis string needs to have exactly 3 orthogonal values from {RUBLDF}!")
        axis_directions = axis_directions.upper()
        base_change = np.identity(4)
        for i, direction in enumerate(axis_directions):
            base_change[:3, i] = CoordinateSystem._axis_char_map[direction]
        if np.linalg.det(base_change) == 0.0:
            raise ValueError(f"{axis_directions} is not a valid coordinate system!")
        return base_change

    def __gt__(self, other: "CoordinateSystem") -> Transformation:
        """Creates a transformation object that can be used to transform coordinates from this coordinate system
        to the given `other` coordinate system.

        Args:
            other (CoordinateSystem): The coordinate system to which coordinates will be transformed.

        Returns:
            Transformation: A Transformation object representing the transformation from this coordinate system
                to the `other` coordinate system.

        Raises:
            ValueError: If the current coordinate system is left-handed.
        """
        base_matrix = self._base_matrix
        if not self.is_right_handed:
            raise ValueError("Left Handed Coordinates Systems are not supported atm!")

        return Transformation.from_transformation_matrix(mat=(other._base_matrix.transpose() @ base_matrix))

    def __lt__(self, other: "CoordinateSystem") -> Transformation:
        """Define the < operator for the class. It returns a Transformation object that represents the
        transformation required to convert the other coordinate system to the current instance of
        CoordinateSystem. This method simply calls the __gt__ method of the other coordinate system

        Args:
            other (CoordinateSystem): The coordinate system to be transformed.

        Returns:
            Transformation: A Transformation object that represents the transformation required to convert
                other coordinate system to the current instance of CoordinateSystem.
        """
        return other > self

    @staticmethod
    def get_base_change_from_to(from_axis_directions: str, to_axis_directions: str) -> Transformation:
        """A method that transforms a vector from the current coordinate system to another coordinate system.

        Args:
            from_axis_directions (str): A string representing a direction in the current coordinate system.
            to_axis_directions (str): A string representing a direction in the target coordinate system.

        Returns:

        """
        return CoordinateSystem(from_axis_directions) > CoordinateSystem(to_axis_directions)

    @staticmethod
    def print_convention():
        """Prints the convention used to define the front, left and up axis of the coordinate system.

        Returns:
            None: This method does not return anything. It only logs the information using the logger module.
        """
        logger.info(f"Front axis: {CoordinateSystem._axis_char_map['F']}")
        logger.info(f"Left axis: {CoordinateSystem._axis_char_map['L']}")
        logger.info(f"Up axis: {CoordinateSystem._axis_char_map['U']}")

    def quaternion_from_rpy(self, roll: float, pitch: float, yaw: float, degrees: bool = False, order: str = "xyz"):
        """Convert the given roll, pitch and yaw angles (in radius or degree) into a quaternion representation.
        The rotation order is specified by the `order` argument.

        Args:
            roll (float): angles (in radius or degree)
            pitch (float): angles (in radius or degree)
            yaw (float): angles (in radius or degree)
            degrees (bool): Whether the roll, pitch, yaw angles are in degree or not. If not, it's in radius.
            order (str): The order for rotation, containing three letters corresponding to the axes
                        of rotation (e.g., "xyz" for roll-pitch-yaw in that order).

        Returns:
            Quaternion: A unit quaternions class to represent rotations in 3D space (w, x, y, z)

        """
        transform = CoordinateSystem("FLU") > self

        front = transform.rotation @ CoordinateSystem._axis_char_map["F"].reshape((3, 1))
        left = transform.rotation @ CoordinateSystem._axis_char_map["L"].reshape((3, 1))
        up = transform.rotation @ CoordinateSystem._axis_char_map["U"].reshape((3, 1))

        rotations = {
            "x": Quaternion(axis=front, radians=roll if not degrees else math.radians(roll)),
            "y": Quaternion(axis=left, radians=pitch if not degrees else math.radians(pitch)),
            "z": Quaternion(axis=up, radians=yaw if not degrees else math.radians(yaw)),
        }
        q = Quaternion()
        for rot in order:
            q = q * rotations[rot]
        return q

    @property
    def forward(self) -> np.ndarray:
        """Returns a numpy array that represents the forward direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the forward direction vector
            of the coordinate system.
        """
        return self._base_matrix[0, :3]

    @property
    def backward(self) -> np.ndarray:
        """Returns a numpy array that represents the backward direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the backward direction vector
            of the coordinate system.
        """
        return -1.0 * self.forward

    @property
    def left(self) -> np.ndarray:
        """Returns a numpy array that represents the left direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the left direction vector
            of the coordinate system.
        """
        return self._base_matrix[1, :3]

    @property
    def right(self) -> np.ndarray:
        """Returns a numpy array that represents the right direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the right direction vector
            of the coordinate system.
        """
        return -1.0 * self.left

    @property
    def up(self) -> np.ndarray:
        """Returns a numpy array that represents the up direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the up direction vector
            of the coordinate system.
        """
        return self._base_matrix[2, :3]

    @property
    def down(self) -> np.ndarray:
        """Returns a numpy array that represents the down direction vector of the coordinate system.

        Returns:
        numpy.ndarray: A numpy array with shape (3,) that represents the down direction vector
            of the coordinate system.
        """
        return -1.0 * self.up

    @staticmethod
    def change_transformation_coordinate_system(
        transformation: Transformation, transformation_system: str, target_system: str
    ) -> Transformation:
        """
        Changes the coordinate system interpretation of a transformation. Note that this only works as intended,
        if the transformation doesn't already include a coordinate system change! For example if you have an
        ego_to_world matrix, going from ego in RFU to world in RFU, you want to know the transformation from ego in FLU
        to world in FLU, you can call
        `CoordinateSystem.change_transformation_coordinate_system(ego_to_world, transformation_system="RFU",
        target_system="FLU")`
        Args:
            transformation: the transformation to change the coordinate system of
            transformation_system: the coordinate system the transformation currently is in
            target_system: the coordinate system to change to

        Returns:
        Transformation: A transformation of the same meaning, but with the changed coordinate system interpretation
        """
        transformation_to_target = CoordinateSystem(transformation_system) > CoordinateSystem(target_system)
        target_to_transformation = transformation_to_target.inverse
        return transformation_to_target @ transformation @ target_to_transformation


INTERNAL_COORDINATE_SYSTEM = CoordinateSystem("FLU")
SIM_COORDINATE_SYSTEM = CoordinateSystem("RFU")
SIM_TO_INTERNAL = SIM_COORDINATE_SYSTEM > INTERNAL_COORDINATE_SYSTEM
