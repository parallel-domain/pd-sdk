from typing import Callable, Optional

import numpy as np

from paralleldomain.data_lab import (
    CustomSimulationAgent,
    CustomSimulationAgentBehavior,
    ExtendedSimState,
    coordinate_system,
)
from paralleldomain.utilities.transformation import Transformation


class LookAtPointBehavior(CustomSimulationAgentBehavior):
    """Assigns a static behavior to an agent which causes it faces towards a particular point in world space

    Args:
        look_from: 3x1 numpy array which contains the x,y,z position from which we look at the target
        look_at: 3x1 numpy array which contains the x,y,z position of the target to be looked at
    """

    def __init__(self, look_from: np.ndarray, look_at: np.ndarray):
        self._look_from = look_from
        self._look_at = look_at
        self._pose: Optional[Transformation] = None  # Initialize only

    # This method assigns a pose to the Custom Agent at the beginning of the scenario
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Calculate a normalized forward vector that defines the direction the Custom Agent should be facing
        desired_direction_vector = self._look_at - self._look_from
        forward_vector = desired_direction_vector / np.linalg.norm(desired_direction_vector)

        # Use the cross product to define a normalized left vector
        left_vector = np.cross(coordinate_system.up, forward_vector)
        left_vector = left_vector / np.linalg.norm(left_vector)

        # Use the left and forward vectors calculated above to calculate an up vector for the Custom Agent's pose
        up_vector = np.cross(forward_vector, left_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        # Create an empty tranformation matrix
        tf = np.eye(4)

        # Populate the transformation matrix with the vectors calculated above
        tf[:3, 0] = -left_vector  # Right
        tf[:3, 1] = forward_vector  # Front
        tf[:3, 2] = up_vector  # Up
        tf[:3, 3] = self._look_from

        # Create a Transformation object that defines the required position of the Custom Agent
        self._pose = Transformation.from_transformation_matrix(mat=tf, approximate_orthogonal=True)

        # Set the Custom Agent to have the pose calculated above
        agent.set_pose(pose=self._pose.transformation_matrix)

    # Since this is a static behavior, there is no need to update the state of the agent at every
    # time step, so this method does nothing.
    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        agent.set_pose(pose=self._pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "LookAtPointBehavior":
        return LookAtPointBehavior(
            look_from=self._look_from,
            look_at=self._look_at,
        )


class StaticBehavior(CustomSimulationAgentBehavior):
    """
    Assigns a static behavior with a given pose to an agent

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
