import logging
import random
from typing import Callable, List, Optional

import numpy as np
from pd.sim import Raycast
from pyquaternion import Quaternion

from paralleldomain import data_lab
from paralleldomain.data_lab import CustomSimulationAgent, CustomSimulationAgentBehavior, ExtendedSimState
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry
from paralleldomain.model.occupancy import OccupancyGrid
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL
from paralleldomain.utilities.geometry import random_point_within_2d_polygon
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


class SingleFrameVehicleBehavior(CustomSimulationAgentBehavior):
    """Controls the placement of a single vehicle in single frame scenarios.  Places the vehicle in a different location
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

    # The method finds a valid pose that the Custom Agent to which this behavior is assigned can be placed
    def _find_new_pose(self, sim_state: ExtendedSimState) -> Transformation:
        map_query = sim_state.map_query

        # Use the MapQuery object to find a random LaneSegment that corresponds to the LaneType specified
        lane_object = map_query.get_random_lane_type_object(
            lane_type=self._lane_type,
            random_seed=self._random_seed,
        )

        # Retrieve the reference line of the found LaneSegment as a numpy array
        reference_line = map_query.get_edge(edge_id=lane_object.reference_line).as_polyline().to_numpy()

        # If the lane segment is defined in the backwards direction, flip it
        if lane_object.direction == LaneSegment.Direction.BACKWARD:
            reference_line = np.flip(reference_line, axis=0)

        # Calculate the difference between the reference line points and extend it by one to retain same length
        diff = np.diff(reference_line, axis=0)
        diff = np.vstack((diff, diff[-1]))

        # Calculate the Transformation required to travel from one point on the reference line to the next.
        # This will be used to place the Custom Agent anywhere along the LaneSegment and ensure that it is aligned
        # with the LaneSegment
        reference_line_transformations = [
            Transformation(
                translation=reference_line[i],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
            )
            for i in range(len(diff))
        ]

        # Choose a point of the LaneSegment that we wish to spawn the CustomAgent beyond
        index_to_spawn_beyond = self._random_state.randint(0, len(reference_line_transformations) - 2)

        # Choose a factor that defines a normalized distance for how far beyond the above point we wish the spawn the
        # Custom Agent on the reference line
        factor = self._random_state.uniform(0.0, 1.0)

        # Interpolate between the two bounding poses on the reference line
        pose = Transformation.interpolate(
            tf0=reference_line_transformations[index_to_spawn_beyond],
            tf1=reference_line_transformations[index_to_spawn_beyond + 1],
            factor=factor,
        )

        return pose

    # This method will set up the state of the Custom Agent at the beginning of the scenario
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Use the above implemented functionality to retrieve a valid pose to place the vehicle
        self._pose = self._find_new_pose(sim_state=sim_state)

        # Set the Agent's pose to the pose found above
        agent.set_pose(pose=self._pose.transformation_matrix)

    # This method will set the state of the Custom Agent at every time step of the scenario
    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # If the current time step corresponds to the time step after a rendered frame, enter this loop
        if (int(sim_state.current_frame_id) % self._sim_capture_rate) == 0 and int(sim_state.current_frame_id) > 0:
            # Increment the random seed
            self._random_seed += 1

            # Use the above implemented functionality to retrieve a valid pose to place the vehicle
            self._pose = self._find_new_pose(sim_state=sim_state)

        # Set the Agent's pose to the pose found above
        agent.set_pose(pose=self._pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "SingleFrameVehicleBehavior":
        return SingleFrameVehicleBehavior(
            lane_type=self._lane_type,
            random_seed=self._random_seed,
        )


class SingleFramePlaceNearEgoBehavior(CustomSimulationAgentBehavior):
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
        align_to_ground_normal: bool = True,
        max_lateral_offset: Optional[float] = None,
        max_rotation_offset_degrees: Optional[float] = None,
        occupancy_check_agent_types: Optional[List] = None,
        max_retries: int = 100,
    ):
        self._random_seed = random_seed
        self._sim_capture_rate = 10
        self._spawn_radius = spawn_radius
        self._lane_type = lane_type
        self._spawn_in_middle_of_lane = (
            True if lane_type == LaneSegment.LaneType.PARKING_SPACE else spawn_in_middle_of_lane
        )
        self._max_lateral_offset = max_lateral_offset
        self._max_rotation_offset_degrees = max_rotation_offset_degrees
        self._occupancy_check_agent_types = occupancy_check_agent_types
        self._max_retries = max_retries
        self._align_to_ground_normal = align_to_ground_normal

        self._pose = None

        self._random_state = random.Random(self._random_seed)

    # This method is responsible for finding a suitable pose for objects to be spawned near the ego vehicle
    def _find_new_pose(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Callable
    ) -> Transformation:
        map_query = sim_state.map_query

        # Use the MapQuery object to search for all lane segments near the ego pose.  This is done by extracting the
        # ego_pose from the sim_state. A list comprehension is then used to keep only the LaneSegments that correspond
        # to the LaneType specified
        lane_objects = [
            lane
            for lane in map_query.get_lane_segments_near(pose=sim_state.ego_pose, radius=self._spawn_radius)
            if lane.type == self._lane_type
        ]

        # If there are no LaneSegments near the ego, this means that the ego has moved to a position in which there are
        # no valid nearby lanes. Raise a warning that the agent to which this behavior is assigned will not be moved
        # and return the current pose of the object so that execution of the scenario can continue
        if len(lane_objects) == 0:
            logger.warning(
                "Ego has moved into a position where it is not possible to place all agents"
                ", some agents will be skipped so specified number of agents may not be respected across"
                " all frames"
            )
            return Transformation.from_transformation_matrix(mat=agent.pose, approximate_orthogonal=True)

        # Sort for determinism
        lane_objects = sorted(lane_objects, key=lambda lane: lane.id)

        lane_valid_spawn_point_map = dict()
        lane_id_weight_map = dict()
        lane_id_lane_object_map = dict()

        for lane in lane_objects:
            reference_line = map_query.get_edge(lane.reference_line).as_polyline().to_numpy()
            reference_line_candidate_points = reference_line[
                np.linalg.norm(reference_line - sim_state.ego_pose.translation, axis=1) < self._spawn_radius
            ]
            if len(reference_line_candidate_points) < 2:
                continue

            distance_between_points = np.linalg.norm(np.diff(reference_line_candidate_points, axis=0), axis=1)
            total_length = np.cumsum(distance_between_points)[-1]

            lane_valid_spawn_point_map[lane.id] = reference_line_candidate_points
            lane_id_weight_map[lane.id] = total_length
            lane_id_lane_object_map[lane.id] = lane

        # If valid lanes are found, enter a while loop to search for valid spawn points
        valid_point_found = False
        attempts = 0
        while not valid_point_found:
            # Randomly choose a lane to spawn, but ensuring we weigh each lane by the length that lies within the max
            #   spawn radis. Otherwise, we risk overloading junctions, etc.
            lane_id_to_spawn = self._random_state.choices(
                list(lane_id_weight_map.keys()), list(lane_id_weight_map.values())
            )[0]
            reference_line = lane_valid_spawn_point_map[lane_id_to_spawn]
            lane_object = lane_id_lane_object_map[lane_id_to_spawn]

            # If the lane is defined backwards, the reference line
            if lane_object.direction == LaneSegment.Direction.BACKWARD:
                reference_line = np.flip(reference_line, axis=0)

            # Calculate the difference between the reference line points and extend it by one to retain same length
            diff = np.diff(reference_line, axis=0)
            diff = np.vstack((diff, diff[-1]))

            # Calculate the transformations to travel from one point to the next on the line. This will be used to
            # place agents aligned with the LaneSegment at all times
            reference_line_transformations = [
                Transformation(
                    translation=reference_line[i],
                    quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[i][0], diff[i][1])),
                )
                for i in range(len(diff))
            ]

            # If the flag is set to not necessarily spawn the Custom Agent in the middle of the lane:
            if not self._spawn_in_middle_of_lane:
                # Choose a point of the LaneSegment that we wish to spawn the CustomAgent beyond
                index_to_spawn_beyond = self._random_state.randint(0, len(reference_line_transformations) - 2)

                # Choose a factor that defines a normalized distance for how far beyond the above point we wish the
                # spawn the Custom Agent on the reference line
                factor = self._random_state.uniform(0.0, 1.0)

                # Interpolate between the two bounding poses on the reference line
                pose = Transformation.interpolate(
                    tf0=reference_line_transformations[index_to_spawn_beyond],
                    tf1=reference_line_transformations[index_to_spawn_beyond + 1],
                    factor=factor,
                )
            else:  # If it is specified that the Custom Agent should be spawned in the middle of the LaneSegment
                # Interpolate between the two bounding poses on the reference line
                pose = Transformation.interpolate(
                    tf0=reference_line_transformations[0],
                    tf1=reference_line_transformations[-1],
                    factor=0.5,
                )

            # Define the bounds around the vehicle the occupancy check should be completed
            # For parked vehicles, we check only the bounding box of the vehicle for occupancy
            # For traffic vehicles, we check 1.5 times the length of the vehicle and the exact width of the vehicle
            # For pedestrians, we check 1.5 times the width of the pedestrian in all directions
            occupancy_length_check_factor_front = 0.5 if self._lane_type == LaneSegment.LaneType.PARKING_SPACE else 1.5
            occupancy_length_check_factor_left = 1.5 if self._lane_type == LaneSegment.LaneType.SIDEWALK else 0.4

            # If lateral offset for spawn location is specified
            if self._max_lateral_offset is not None:
                # Create a vector that defines the direction in which the lateral offset will be applied.  Note that
                # this is calculated through matrix multiplication with the pose's rotation matrix. Also note we
                # randomly choose whether offset to the left or right
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
            else:  # If no lateral offset specified, no offset applied
                offset_translation = pose.translation

            # If rotation offset for spawn location is specified
            if self._max_rotation_offset_degrees is not None:
                # Randomly choose the magnitude and direction of the rotation
                rotation_magnitude = self._random_state.choice([-1.0, 1.0]) * self._random_state.uniform(
                    0, self._max_rotation_offset_degrees
                )

                # Create a Transformation object that defines the rotation to apply
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

            # Create the spawn pose transformation object, that includes all lateral and rotation offsets applied above
            # (if applicable)
            spawn_pose = Transformation.from_euler_angles(
                angles=offset_rotation, order="xyz", translation=offset_translation
            )

            # Calculate vectors which define the forward and left directions of the agent. These will be used to
            # perform occupancy checks
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

            # Collect the points that define the bounds of the region as well as the filled points into one array
            points_to_check = np.vstack((filled_points, points_to_check))

            # If the spot is not occupied, we exit the loop.
            # The occupancy check is performed using the is_occupied_world() method of the occupancy_grid attribute in
            # the sim_state

            if self._occupancy_check_agent_types is not None:
                vehicle_boxes = [
                    BoundingBox3DGeometry(
                        # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU
                        pose=Transformation.from_transformation_matrix(a.pose, approximate_orthogonal=True)
                        @ SIM_TO_INTERNAL.inverse,
                        width=a.width,
                        height=a.height,
                        length=a.length,
                    )
                    for a in sim_state.current_agents
                    if isinstance(a.step_agent, tuple(self._occupancy_check_agent_types))
                ]
                occupancy_grid = OccupancyGrid.from_bounding_boxes_3d(boxes=vehicle_boxes)

            else:
                occupancy_grid = sim_state.current_occupancy_grid

            if (~occupancy_grid.is_occupied_world(points=points_to_check)).all():
                valid_point_found = True

            # If the maximum number of search attempts has been exceeded, raise a warning that a valid spawn pose
            # cannot be found and return the current pose so the Custom Agent does not get moved.  This is done to
            # prevent blocking execution
            elif attempts > self._max_retries:
                logger.warning(
                    "Ego has moved into a position where it is not possible to place all agents"
                    ", some agents will be skipped so specified number of agents may not be respected across"
                    " all frames"
                )
                return Transformation.from_transformation_matrix(mat=agent.pose, approximate_orthogonal=True)
            else:  # If valid spawn not found, do the loop again
                attempts += 1

        return self._orient_to_ground(raycast=raycast, current_pose=spawn_pose)

    def _orient_to_ground(self, raycast: Callable, current_pose: Transformation):
        current_xyz = current_pose.translation
        ray_start = current_xyz + data_lab.coordinate_system.up * 10
        result = raycast(
            [Raycast(origin=tuple(ray_start), direction=tuple(data_lab.coordinate_system.down), max_distance=30)]
        )
        ground_height = result[0][0].position[2]
        normal = result[0][0].normal

        # Align to ground height
        new_xyz = np.append(current_xyz[:2], ground_height)
        new_pose = Transformation(quaternion=current_pose.quaternion, translation=new_xyz)

        box_rotation_matrix = current_pose.quaternion.rotation_matrix

        if self._align_to_ground_normal:
            rotation_axis = np.cross(np.array(data_lab.coordinate_system.up), np.array(normal))

            # This check ensures the normal is not exactly equal to the up direction.
            if np.any(rotation_axis):
                rotation_angle = np.arccos(np.dot(data_lab.coordinate_system.up, normal) / np.linalg.norm(normal))
                rotation = Transformation.from_axis_angle(axis=rotation_axis, angle=rotation_angle)
                rotation_matrix = rotation.quaternion.rotation_matrix
                aligned_rotation_matrix = np.dot(rotation_matrix, box_rotation_matrix)
                new_pose = Transformation(translation=new_xyz, quaternion=Quaternion(matrix=aligned_rotation_matrix))

        return new_pose

    # This method will set up the state of the Custom Agent at the beginning of the scenario
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Increment the seed
        self._random_seed += 1

        # Use the above implemented functionality to retrieve a valid pose to place the vehicle
        self._pose = self._find_new_pose(sim_state=sim_state, agent=agent, raycast=raycast)

        # Set the Agent's pose to the pose found above
        agent.set_pose(pose=self._pose.transformation_matrix)

    # This method will set the state of the Custom Agent at every time step of the scenario
    def update_state(
        self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, raycast: Optional[Callable] = None
    ):
        # This assumes 5 start skip frames, sim capture rate of 10, and that we want to "move" 5 frames after each
        # capture frame.
        if (int(sim_state.current_frame_id) % self._sim_capture_rate) == 0 and int(sim_state.current_frame_id) > 0:
            # Increment the random seed
            self._random_seed += 1

            # Use the above implemented functionality to retrieve a valid pose to place the vehicle
            self._pose = self._find_new_pose(sim_state=sim_state, agent=agent, raycast=raycast)

        # Set the Agent's pose to the pose found above
        agent.set_pose(pose=self._pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "SingleFramePlaceNearEgoBehavior":
        return SingleFramePlaceNearEgoBehavior(
            random_seed=self._random_seed,
            spawn_radius=self._spawn_radius,
            spawn_in_middle_of_lane=self._spawn_in_middle_of_lane,
            max_lateral_offset=self._max_lateral_offset,
            max_rotation_offset_degrees=self._max_rotation_offset_degrees,
            max_retries=self._max_retries,
            lane_type=self._lane_type,
            align_to_ground_normal=self._align_to_ground_normal,
            occupancy_check_agent_types=self._occupancy_check_agent_types,
        )
