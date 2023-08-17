from typing import Optional, Callable

import numpy as np
from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2
from pyquaternion import Quaternion

from paralleldomain.data_lab import (
    ExtendedSimState,
    CustomSimulationAgent,
    CustomSimulationAgentBehaviour,
    coordinate_system,
)
from paralleldomain.utilities import inherit_docs
from paralleldomain.utilities.transformation import Transformation


@inherit_docs
class VehicleBehavior(pd_unified_generator_pb2.VehicleBehavior):
    ...


@inherit_docs
class PedestrianBehavior(pd_unified_generator_pb2.PedestrianBehavior):
    ...


@inherit_docs
class Gear(pd_unified_generator_pb2.Gear):
    ...


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


class RenderEgoBehavior(CustomSimulationAgentBehaviour):
    """
    When used in conjunction with RenderEgoGenerator, allows the Ego Vehicle to be rendered
    """

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        initial_pose = sim_state.ego_pose

        agent.set_pose(pose=initial_pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        current_pose = sim_state.ego_pose

        agent.set_pose(pose=current_pose.transformation_matrix)

    def clone(self) -> "RenderEgoBehavior":
        return RenderEgoBehavior()


class DrivewayCreepBehavior(CustomSimulationAgentBehaviour):
    """
    Controls the behavior of vehicles that perform the driveway creeping behavior

    Args:
        reference_line:
              Description:
                  The driveway reference line along which the vehicle should perform the creeping behavior
              Range:
                  n x 3 numpy array containing n points along the reference line.
                  Each point is the [x, y, z] location of that point
              Required:
                  Yes, unless used in conjunction with DrivewayCreepGenerator.
                  When DrivewayCreepBehavior is used in conjunction with DrivewayCreepGenerator,
                  this field is populated automatically
        behavior_duration:
              Description:
                  Length of time in seconds in which the vehicle should travel from the start to end of the driveway
              Required:
                  Yes, unless used in conjunction with DrivewayCreepGenerator.
                  When DrivewayCreepBehavior is used in conjunction with DrivewayCreepGenerator,
                  this field is populated automatically
        agent_length:
              Description:
                  Length of the agent (in meters) to which we are assigning the Driveway Creeping Behavior
              Required:
                  No, if not provided, a default length of 5.5m is used.
                  When DrivewayCreepBehavior is used in conjunction with DrivewayCreepGenerator,
                  this field is populated automatically
    """

    def __init__(self, reference_line: np.ndarray, behavior_duration: int, agent_length: float = 5.5):
        self._behavior_duration = behavior_duration
        self._agent_length = agent_length  # Meters
        # self._reference_line = reference_line
        self._pose: Optional[Transformation] = None  # Initialize only

        # Diff the reference points but extend it, so we can calculate the final pose
        diff = np.diff(reference_line, axis=0)
        diff = np.vstack((diff, diff[-1]))

        self._driveway_transformations = [
            Transformation(
                translation=reference_line[i],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
            )
            for i in range(len(diff))
        ]

        # Cumulative distances between driveway reference points (prepend a 0 to maintain array length)
        self._cumulative_distances = np.insert(
            np.cumsum(np.linalg.norm(np.diff(reference_line, axis=0), axis=1)), 0, 0.0
        )

    def _find_pose(self, meters_travelled: float) -> Transformation:
        bounding_indicies = next(
            (
                (i, i - 1)
                for i in range(len(self._cumulative_distances))
                if self._cumulative_distances[i] > meters_travelled
            ),
            (len(self._cumulative_distances) - 1, len(self._cumulative_distances) - 2),
        )

        factor = (meters_travelled - self._cumulative_distances[bounding_indicies[1]]) / (
            self._cumulative_distances[bounding_indicies[0]] - self._cumulative_distances[bounding_indicies[1]]
        )

        interpolated_pose = Transformation.interpolate(
            tf0=self._driveway_transformations[bounding_indicies[1]],
            tf1=self._driveway_transformations[bounding_indicies[0]],
            factor=factor,
        )

        return interpolated_pose

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        initial_pose = self._find_pose(meters_travelled=self._agent_length / 2)  # Offset to have vehicles start flush

        agent.set_pose(pose=initial_pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        meters_travelled = (sim_state.sim_time / self._behavior_duration) * (
            self._cumulative_distances[-1] - self._agent_length
        ) + (self._agent_length / 2)

        interpolated_pose = self._find_pose(meters_travelled=meters_travelled)

        agent.set_pose(pose=interpolated_pose.transformation_matrix)

    def clone(self) -> "DrivewayCreepBehavior":
        return DrivewayCreepBehavior(
            reference_line=np.array([o.translation for o in self._driveway_transformations]),
            behavior_duration=self._behavior_duration,
            agent_length=self._agent_length,
        )
