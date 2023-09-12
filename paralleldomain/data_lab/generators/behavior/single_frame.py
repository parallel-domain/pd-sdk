import random
from typing import Optional, Callable

import numpy as np
from pd.core import PdError
from pyquaternion import Quaternion

from paralleldomain.data_lab import CustomSimulationAgentBehaviour, ExtendedSimState, CustomSimulationAgent
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.utilities.geometry import random_point_within_2d_polygon
from paralleldomain.utilities.transformation import Transformation


class SingleFrameVehicleBehavior(CustomSimulationAgentBehaviour):
    """
    Controls the placement of a single vehicle in single frame scenarios.  Places the vehicle in a different location
        on the map during every rendered frame

    Args:
        lane_type: The lane type on which the vehicle should be spawned
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
    """

    def __init__(self, lane_type: LaneSegment.LaneType, random_seed: int):
        self._lane_type = lane_type
        self._random_seed = random_seed
        self._sim_capture_rate = 10

        self._pose = None

        self._random_state = random.Random(self._random_seed)

    def _find_new_pose(self, sim_state: ExtendedSimState) -> Transformation:
        # Get a lane object which corresponds to the lane type specified
        lane_object = sim_state.map_query.get_random_lane_type_object(
            lane_type=self._lane_type,
            random_seed=self._random_seed,
        )

        # Pull the reference line of the lane object
        reference_line = sim_state.map.edges[lane_object.reference_line].as_polyline().to_numpy()

        # If the lane is defined backwards, flip it
        if lane_object.direction == LaneSegment.Direction.BACKWARD:
            reference_line = np.flip(reference_line, axis=0)

        # Calculate the difference between the reference line points and extend it by one to retain same length
        diff = np.diff(reference_line, axis=0)
        diff = np.vstack((diff, diff[-1]))

        # Calculate the transformations to travel from one point to the next on the line
        reference_line_transformations = [
            Transformation(
                translation=reference_line[i],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
            )
            for i in range(len(diff))
        ]

        # Choose an index of the random line to spawn beyond
        index_to_spawn_beyond = self._random_state.randint(0, len(reference_line_transformations) - 2)
        factor = self._random_state.uniform(0.0, 1.0)
        pose = Transformation.interpolate(
            tf0=reference_line_transformations[index_to_spawn_beyond],
            tf1=reference_line_transformations[index_to_spawn_beyond + 1],
            factor=factor,
        )

        return pose

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        self._pose = self._find_new_pose(sim_state=sim_state)

        agent.set_pose(pose=self._pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # Change location right after capture for antialiasing
        if (int(sim_state.current_frame_id) % self._sim_capture_rate) - 5 == 1:
            self._random_seed += 1
            self._pose = self._find_new_pose(sim_state=sim_state)

        agent.set_pose(pose=self._pose.transformation_matrix)

    def clone(self) -> "SingleFrameVehicleBehavior":
        return SingleFrameVehicleBehavior(
            lane_type=self._lane_type,
            random_seed=self._random_seed,
        )


class SingleFramePlaceNearEgoBehavior(CustomSimulationAgentBehaviour):
    """
    Controls the placement of a single vehicle in single frame scenarios.  Places the vehicle in a different location
        on the map during every rendered frame

    Args:
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        spawn_radius: The radius (in meters) of the valid spawn region in which the agent can be placed around the ego
            agent
        lane_type: The lane type on which the agent should be spawned
        spawn_in_middle_of_lane: If True, the agent will be spawned in the middle of a randomly chosen lane segment.
            If False, the agent will be spawned at a random point along the randomly chosen lane segment
        max_lateral_offset: The maximum distance (in meters) that the placed agent will be laterally offset from the
            center of the randomly chosen lane segment
        max_rotation_offset_degrees: The maximum rotation (in degrees) that the placed agent will be rotated relative
            to the center line of the randomly chosen lane segment
        max_retries: The maximum number of times to attempt finding a valid spawn location

    Raises:
        PdError: If a valid spawn location is not found within the specified max_retries number of attempts
    """

    def __init__(
        self,
        random_seed: int,
        spawn_radius: float,
        lane_type: LaneSegment.LaneType,
        spawn_in_middle_of_lane: bool = False,
        max_lateral_offset: Optional[float] = None,
        max_rotation_offset_degrees: Optional[float] = None,
        max_retries: int = 1000,
    ):
        self._random_seed = random_seed
        self._sim_capture_rate = 10
        self._spawn_radius = spawn_radius
        self._lane_type = lane_type
        self._spawn_in_middle_of_lane = (
            True if lane_type is LaneSegment.LaneType.PARKING_SPACE else spawn_in_middle_of_lane
        )
        self._max_lateral_offset = max_lateral_offset
        self._max_rotation_offset_degrees = max_rotation_offset_degrees
        self._max_retries = max_retries

        self._pose = None

        self._random_state = random.Random(self._random_seed)

    def _find_new_pose(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent) -> Transformation:
        # Find all the lane objects in the valid spawn region corresponding to the specified lane type
        lane_objects = [
            lane
            for lane in sim_state.map_query.get_lane_segments_near(pose=sim_state.ego_pose, radius=self._spawn_radius)
            if lane.type is self._lane_type
        ]

        valid_point_found = False
        attempts = 0

        # Keep looping while valid spawn not found
        while not valid_point_found:
            # Randomly choose a lane object and get the reference line
            lane_object = self._random_state.choice(lane_objects)
            reference_line = sim_state.map.edges[lane_object.reference_line].as_polyline().to_numpy()

            # If the lane is defined backwards, flip it
            if lane_object.direction == LaneSegment.Direction.BACKWARD:
                reference_line = np.flip(reference_line, axis=0)

            # Calculate the difference between the reference line points and extend it by one to retain same length
            diff = np.diff(reference_line, axis=0)
            diff = np.vstack((diff, diff[-1]))

            # Calculate the transformations to travel from one point to the next on the line
            reference_line_transformations = [
                Transformation(
                    translation=reference_line[i],
                    quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
                )
                for i in range(len(diff))
            ]

            if not self._spawn_in_middle_of_lane:  # If we don't want to spawn in the middle of the reference line
                # Choose a random location between two random points on the line to spawn
                index_to_spawn_beyond = self._random_state.randint(0, len(reference_line_transformations) - 2)
                factor = self._random_state.uniform(0.0, 1.0)
                pose = Transformation.interpolate(
                    tf0=reference_line_transformations[index_to_spawn_beyond],
                    tf1=reference_line_transformations[index_to_spawn_beyond + 1],
                    factor=factor,
                )
            else:  # If we want to spawn in the middle of the reference line
                pose = Transformation.interpolate(
                    tf0=reference_line_transformations[0],
                    tf1=reference_line_transformations[-1],
                    factor=0.5,
                )

            # Define the bounds around the vehicle the occupancy check should be completed
            # For parked vehicles, we check only the bounding box of the vehicle for occupancy
            # For traffic vehicles, we check 1.5 times the length of the vehicle and the exact width of the vehicle
            # For pedestrians, we check 1.5 times the width of the pedestrian in all directions
            occupancy_length_check_factor_front = 0.5 if self._lane_type is LaneSegment.LaneType.PARKING_SPACE else 1.5
            occupancy_length_check_factor_left = 1.5 if self._lane_type is LaneSegment.LaneType.SIDEWALK else 0.5

            # If lateral offset is specified
            if self._max_lateral_offset is not None:
                # Randomly choose whether offset to the left or right
                offset_direction = (
                    self._random_state.choice([-1.0, 1.0])
                    * pose.quaternion.rotation_matrix
                    @ Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
                    @ np.array([0, 1, 0])
                )

                # Choose the magnitude of the offset randomly and apply it to the translation
                offset_translation = pose.translation + self._random_state.uniform(
                    0, self._max_lateral_offset
                ) * offset_direction / np.linalg.norm(offset_direction)
            else:  # If no lateral offset specified, no offset pllied
                offset_translation = pose.translation

            # If rotation offset is specified
            if self._max_rotation_offset_degrees is not None:
                # Randomly choose the magnitude of the rotation
                rotation_magnitude = self._random_state.choice([-1.0, 1.0]) * self._random_state.uniform(
                    0, self._max_rotation_offset_degrees
                )

                # Apply the rotation and store it
                rotated_pose = Transformation.from_transformation_matrix(
                    mat=(
                        pose.transformation_matrix
                        @ Quaternion(axis=[0, 0, 1], radians=np.deg2rad(rotation_magnitude)).transformation_matrix
                    )
                )
                offset_rotation = rotated_pose.as_euler_angles(order="xyz")

            else:
                # If no rotation is specified, no rotation offset applied
                offset_rotation = pose.as_euler_angles(order="xyz")

            # Create the spawn pose transformation object
            spawn_pose = Transformation.from_euler_angles(
                angles=offset_rotation, order="xyz", translation=offset_translation
            )

            # Calculate vectors which define the forward and left directions of the agent
            agent_front_direction = (
                occupancy_length_check_factor_front
                * agent.length
                * spawn_pose.quaternion.rotation_matrix
                @ np.array([0, 1, 0])
            )
            agent_left_direction = (
                occupancy_length_check_factor_left
                * agent.width
                * spawn_pose.quaternion.rotation_matrix
                @ Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
                @ np.array([0, 1, 0])
            )

            # Define the four corners of the bounding box which defines the region we need to check for occupancy
            points_to_check = np.array(
                [
                    spawn_pose.translation + agent_front_direction + agent_left_direction,  # Front left
                    spawn_pose.translation + agent_front_direction - agent_left_direction,  # Front right
                    spawn_pose.translation - agent_front_direction + agent_left_direction,  # Back left
                    spawn_pose.translation - agent_front_direction - agent_left_direction,  # Back right
                ]
            )[:, :2]

            # Fill the occupancy check region with points for the occupancy check
            filled_points = random_point_within_2d_polygon(
                edge_2d=points_to_check, random_seed=self._random_seed, num_points=10
            )
            points_to_check = np.vstack((filled_points, points_to_check))

            # If the spot is not occupied, we pass
            if (~sim_state.current_occupancy_grid.is_occupied_world(points=points_to_check)).all() and np.linalg.norm(
                spawn_pose.translation - sim_state.ego_pose.translation
            ) < self._spawn_radius:
                valid_point_found = True
            elif attempts > self._max_retries:
                raise PdError(
                    "Could not find valid spawn location for vehicle.  "
                    "Try reducing number of vehicles or increasing spawn radius"
                )
            else:  # If valid spawn not found, do the loop again
                attempts += 1

        return spawn_pose

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        self._random_seed += 1

        self._pose = self._find_new_pose(sim_state=sim_state, agent=agent)

        agent.set_pose(pose=self._pose.transformation_matrix)

    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # Change location right after capture for antialiasing
        if (int(sim_state.current_frame_id) % self._sim_capture_rate) - 5 == 1:
            self._random_seed += 1
            self._pose = self._find_new_pose(sim_state=sim_state, agent=agent)

        agent.set_pose(pose=self._pose.transformation_matrix)

    def clone(self) -> "SingleFramePlaceNearEgoBehavior":
        return SingleFramePlaceNearEgoBehavior(
            random_seed=self._random_seed,
            spawn_radius=self._spawn_radius,
            spawn_in_middle_of_lane=self._spawn_in_middle_of_lane,
            max_lateral_offset=self._max_lateral_offset,
            max_rotation_offset_degrees=self._max_rotation_offset_degrees,
            max_retries=self._max_retries,
            lane_type=self._lane_type,
        )
