from typing import Callable, Optional

import numpy as np
from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2
from pyquaternion import Quaternion

from paralleldomain.data_lab import CustomSimulationAgent, CustomSimulationAgentBehavior, ExtendedSimState
from paralleldomain.utilities import inherit_docs
from paralleldomain.utilities.transformation import Transformation

Gear = pd_unified_generator_pb2.Gear


@inherit_docs
class VehicleBehavior(pd_unified_generator_pb2.VehicleBehavior):
    ...


class RenderEgoBehavior(CustomSimulationAgentBehavior):
    """When used in conjunction with RenderEgoGenerator, allows the Ego Vehicle to be rendered"""

    # The premise of rendering the ego vehicle is very simple.  Since the sim_state already contains the pose of the ego
    # in the scene, we simply need to adjust the pose of this agent to align with the pose of the ego at every time
    # step.  Thus, both set_initial_state() and update_state() will retrieve the ego_pose from the sim_state and assign
    # that pose to the Custom Agent to which this behavior is assigned
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Retrieve the ego_pose from the sim_state
        initial_pose = sim_state.ego_pose

        # Set the pose of the Custom Agent to the ego_pose
        agent.set_pose(pose=initial_pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # Retrieve the ego_pose from the sim_state
        current_pose = sim_state.ego_pose

        # Set the pose of the Custom Agent to the ego_pose
        agent.set_pose(pose=current_pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "RenderEgoBehavior":
        return RenderEgoBehavior()


class DrivewayCreepBehavior(CustomSimulationAgentBehavior):
    """Controls the behavior of vehicles that perform the driveway creeping behavior

    Args:
        reference_line: The driveway reference line along which the vehicle should perform the creeping behavior.
            Numpy array should be n x 3 in size containing n points along the reference line where each point is the
            [x, y, z] location of that point
        behavior_duration:
            Length of time in seconds in which the vehicle should travel from the start to end of the driveway
        agent_length:
            Length of the agent (in meters) to which we are assigning the Driveway Creeping Behavior
    """

    def __init__(self, reference_line: np.ndarray, behavior_duration: float, agent_length: float = 5.5):
        # Store parameters passed into the function so that they can be accessed by other methods
        self._behavior_duration = behavior_duration
        self._agent_length = agent_length  # Meters
        self._pose: Optional[Transformation] = None  # Initialize only

        # Calculate the difference in positions between the points that define the reference line of the driveway
        diff = np.diff(reference_line, axis=0)

        # Copy the last value of the array so that the diff array is the same size as the reference line array
        diff = np.vstack((diff, diff[-1]))

        # Use the diff array and the reference line, to create a list of poses that the vehicle should be in at every
        # point of the reference line.  This ensures that the vehicle will be aligned with the driveway curvature at
        # all times
        self._driveway_transformations = [
            Transformation(
                translation=reference_line[i],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
            )
            for i in range(len(diff))
        ]

        # Calculate cumulative distances between driveway reference points (prepend a 0 to maintain array length)
        self._cumulative_distances = np.insert(
            np.cumsum(np.linalg.norm(np.diff(reference_line, axis=0), axis=1)), 0, 0.0
        )

    # This method is a helper function which takes in the meters a vehicle has travelled along a reference line, and
    # automatically calculates the pose the vehicle should be in.  It automatically determines which two poses in the
    # reference line to interpolate between
    def _find_pose(self, meters_travelled: float) -> Transformation:
        # Calculate the indices of the points on the reference line which sit on either side of the vehicle
        bounding_indices = next(
            (
                (i, i - 1)
                for i in range(len(self._cumulative_distances))
                if self._cumulative_distances[i] > meters_travelled
            ),
            (len(self._cumulative_distances) - 1, len(self._cumulative_distances) - 2),
        )

        # Calculate a normalized factor of how far the vehicle has travelled between the reference points that bound
        # the vehicles location along the reference line
        factor = (meters_travelled - self._cumulative_distances[bounding_indices[1]]) / (
            self._cumulative_distances[bounding_indices[0]] - self._cumulative_distances[bounding_indices[1]]
        )

        # Perform interpolation of the poses to find the exact pose the vehicle should be in for how far along the
        # driveway it has travelled
        interpolated_pose = Transformation.interpolate(
            tf0=self._driveway_transformations[bounding_indices[1]],
            tf1=self._driveway_transformations[bounding_indices[0]],
            factor=factor,
        )

        # Return the interpolated pose that was calculated above
        return interpolated_pose

    # This method is responsible for setting the starting position of the vehicle to which we assign this behavior.
    # In this case, we simply set the vehicles pose to that at the beginning of the driveway
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # We calculate the pose of the vehicle when it has travelled half of its length along the driveway
        # so that the vehicle does not overhang the edge of the driveway
        initial_pose = self._find_pose(meters_travelled=self._agent_length / 2)

        # Set the agents pose to the pose found above
        agent.set_pose(pose=initial_pose.transformation_matrix)

    # This method is called every time a simulation time step is advanced.  As such, any logic for how an agent moves
    # throughout a scenario should be implemented in this method.

    # In this case, this method is responsible for moving the vehicle along the driveway as the scenario progresses
    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # Calculate the meters along the driveway the vehicle has travelled based on the sim_time of the sim_state
        meters_travelled = (sim_state.sim_time / self._behavior_duration) * (
            self._cumulative_distances[-1] - self._agent_length
        ) + (self._agent_length / 2)

        # Use the helper function to find the pose the vehicle should be at
        interpolated_pose = self._find_pose(meters_travelled=meters_travelled)

        # Set the vehicle's pose to that found above
        agent.set_pose(pose=interpolated_pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "DrivewayCreepBehavior":
        return DrivewayCreepBehavior(
            reference_line=np.array([o.translation for o in self._driveway_transformations]),
            behavior_duration=self._behavior_duration,
            agent_length=self._agent_length,
        )
