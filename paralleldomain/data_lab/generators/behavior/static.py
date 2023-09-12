from typing import Optional, Callable

import numpy as np

from paralleldomain.data_lab import (
    CustomSimulationAgentBehaviour,
    ExtendedSimState,
    CustomSimulationAgent,
    coordinate_system,
)
from paralleldomain.utilities.transformation import Transformation


class LookAtPointBehavior(CustomSimulationAgentBehaviour):
    """
    Allows a static agent to be created that faces towards a particular point in world space

    Args:
        look_from: 3x1 numpy array which contains the x,y,z position from which we look at the target
        look_at: 3x1 numpy array which contains the x,y,z position of the target to be looked at
    """

    def __init__(self, look_from: np.ndarray, look_at: np.ndarray):
        self._look_from = look_from
        self._look_at = look_at
        self._pose: Optional[Transformation] = None  # Initialize only

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        desired_direction_vector = self._look_at - self._look_from
        forward_vector = desired_direction_vector / np.linalg.norm(desired_direction_vector)

        left_vector = np.cross(coordinate_system.up, forward_vector)
        left_vector = left_vector / np.linalg.norm(left_vector)

        up_vector = np.cross(forward_vector, left_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        tf = np.eye(4)

        tf[:3, 0] = -left_vector  # Right
        tf[:3, 1] = forward_vector  # Front
        tf[:3, 2] = up_vector  # Up
        tf[:3, 3] = self._look_from

        self._pose = Transformation.from_transformation_matrix(mat=tf, approximate_orthogonal=True)

        agent.set_pose(pose=self._pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        agent.set_pose(pose=self._pose.transformation_matrix)

    def clone(self) -> "LookAtPointBehavior":
        return LookAtPointBehavior(
            look_from=self._look_from,
            look_at=self._look_at,
        )


class StaticBehavior(CustomSimulationAgentBehaviour):
    """
    Allows a static agent to be created with a given pose

    Args:
        pose: The pose that should be maintained throughout the scenario.  This pose
            will not change throughout the scenario.
    """

    def __init__(self, pose: Transformation):
        self._pose = pose

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        agent.set_pose(pose=self._pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        agent.set_pose(pose=self._pose.transformation_matrix)

    def clone(self) -> "StaticBehavior":
        return StaticBehavior(pose=self._pose)
