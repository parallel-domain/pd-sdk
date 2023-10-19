import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from pd.core import PdError
from pyquaternion import Quaternion

from paralleldomain.data_lab import (
    CustomSimulationAgent,
    CustomSimulationAgentBehavior,
    ExtendedSimState,
    coordinate_system,
)
from paralleldomain.data_lab.config.map import LaneSegment, RoadSegment, Side
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.utilities.transformation import Transformation


class TrafficSignPoleBehavior(CustomSimulationAgentBehavior):
    """Behavior that spawns traffic sign poles to which traffic signs can be attached.  This behavior works by defining
        a region in which traffic sign poles can be placed, and randomly selecting points within that region to spawn
        traffic sign poles

    Args:
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        radius: The radius of the region in which traffic signs should be spawned
        single_frame_mode: Flag to indicate whether this generator is being used to configure a single frame
            dataset. In single frame datasets, the location of the vehicles and the signage changes between each
            rendered frame
        forward_offset_to_place_signs: The distance (in meters) in front of the ego vehicle on which the region in
            which traffic signs can be spawned should be centered
        min_distance_between_signs: The minimum distance (in meters) between traffic poles in the scenario
        max_retries: The maximum number of times the behavior will attempt to find a valid spawn location for the
            traffic sign pole

    Raises:
        PdError: If a valid spawn location cannot be found within the specified max_retries
    """

    def __init__(
        self,
        random_seed: int,
        radius: float,
        single_frame_mode: bool,
        orient_signs_facing_travel_direction: bool,
        forward_offset_to_place_signs: float = 15,
        min_distance_between_signs: float = 2.5,
        max_retries: int = 100,
    ):
        self._random_seed = random_seed
        self._radius = radius
        self._single_frame_mode = single_frame_mode
        self._orient_signs_facing_travel_direction = orient_signs_facing_travel_direction
        self._forward_offset_to_place_signs = forward_offset_to_place_signs
        self._min_distance_between_signs = min_distance_between_signs
        self._max_retries = max_retries

        self._sim_capture_rate = 10  # Enforced for single frames, not needed for temporal scenes
        self._random_state = random.Random(self._random_seed)

    # The internal function is used to find a valid place to spawn a traffic pole
    def _find_spawn_point(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent) -> Transformation:
        # The first thing we must do is define a valid search region centered the specified distance ahead of the ego
        # vehicle

        # Define a vector which denotes how far in front of the ego the center of the valid spawn region should be. To
        # do this, we use the ego_pose in the sim_state
        ego_forward_direction = (
            self._forward_offset_to_place_signs * sim_state.ego_pose.quaternion.rotation_matrix @ np.array([0, 1, 0])
        )

        # Store the center of the valid search region and create a Transformation object that includes the rotation
        # of the ego that we retrieve from the sim_state
        search_center_location = sim_state.ego_pose.translation + ego_forward_direction
        search_center_pose = Transformation.from_euler_angles(
            angles=sim_state.ego_pose.as_euler_angles(order="xyz", degrees=False),
            order="xyz",
            degrees=False,
            translation=search_center_location,
        )

        invalid_road_types = {
            RoadSegment.RoadType.UNCLASSIFIED,
            RoadSegment.RoadType.DRIVEWAY,
            RoadSegment.RoadType.PARKING_AISLE,
            RoadSegment.RoadType.DRIVEWAY_PARKING_ENTRY,
        }

        bounds = BoundingBox2DBaseGeometry(
            x=search_center_pose.translation[0] - self._radius,
            y=search_center_pose.translation[1] - self._radius,
            width=2 * self._radius,
            height=2 * self._radius,
        )

        roads_near = [
            road
            for road in sim_state.map_query.get_road_segments_within_bounds(bounds=bounds, method="overlap")
            if road.type not in invalid_road_types
        ]

        # Sort for determinism
        roads_near = sorted(roads_near, key=lambda r: r.id)

        # Create a dictionary mapping road id to a list of candidate ref line points for that roads left and right edge
        # This is to avoid bias in spawning.
        road_valid_spawn_point_map = defaultdict(dict)
        road_id_weight_map = dict()
        for road in roads_near:
            right_edge = sim_state.map_query.get_edge_of_road_from_lane(road.lane_segments[0], side=Side.RIGHT)
            left_edge = sim_state.map_query.get_edge_of_road_from_lane(road.lane_segments[0], side=Side.LEFT)
            right_edge_ref_line = right_edge.as_polyline().to_numpy()
            left_edge_ref_line = left_edge.as_polyline().to_numpy()

            right_edge_candidate_points = right_edge_ref_line[
                np.linalg.norm(right_edge_ref_line - search_center_location, axis=1) < self._radius
            ]
            left_edge_candidate_points = left_edge_ref_line[
                np.linalg.norm(left_edge_ref_line - search_center_location, axis=1) < self._radius
            ]

            road_valid_spawn_point_map[road.id][Side.LEFT] = left_edge_candidate_points
            road_valid_spawn_point_map[road.id][Side.RIGHT] = right_edge_candidate_points
            road_id_weight_map[road.id] = len(left_edge_candidate_points) + len(right_edge_candidate_points)

        if len(roads_near) == 0:
            raise PdError(
                "Ego location has no valid places to spawn signs. Adjust ego spawn "
                "location or increase radius to spawn signs in"
            )

        valid_spawn_found = False
        attempts = 0
        while not valid_spawn_found:
            road_to_spawn_on = self._random_state.choices(
                list(road_id_weight_map.keys()), list(road_id_weight_map.values())
            )[0]
            side_to_spawn_on = Side.LEFT if self._random_state.uniform(0, 1) < 0.5 else Side.RIGHT
            edge_line_to_spawn = road_valid_spawn_point_map[road_to_spawn_on][side_to_spawn_on]

            if edge_line_to_spawn.shape[0] < 2:
                # To avoid failure for now
                attempts += 1
                continue

            # Randomly choose where along the line to space
            index_to_spawn_beyond = self._random_state.randint(0, len(edge_line_to_spawn) - 2)
            factor = self._random_state.uniform(0.0, 1.0)

            # Calculate the diff between the preceding and succeeding point on the line
            diff = edge_line_to_spawn[index_to_spawn_beyond + 1, :] - edge_line_to_spawn[index_to_spawn_beyond]

            # Create a pose containing the potential spawn location of the pole
            spawn_pose = Transformation(
                translation=factor * diff + edge_line_to_spawn[index_to_spawn_beyond, :],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[0], diff[1])),
            )

            # Find the left and forward directions from the pole (incorporating the min_distance_between_signs).  This
            # will be used to define a bounding box for the sign in which no other agents should be located
            pole_front_direction = (
                self._min_distance_between_signs * spawn_pose.quaternion.rotation_matrix @ np.array([0, 1, 0])
            )
            pole_left_direction = (
                agent.width
                / 2
                * spawn_pose.quaternion.rotation_matrix
                @ Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
                @ np.array([0, 1, 0])
            )

            # Using the above vectors, we create a region that bounds the region around the sign pole in which no other
            # signs should be located. We check just the four corners to make sure they aren't occupied
            points_to_check = np.array(
                [
                    spawn_pose.translation + pole_front_direction + pole_left_direction,
                    spawn_pose.translation + pole_front_direction - pole_left_direction,
                    spawn_pose.translation - pole_front_direction + pole_left_direction,
                    spawn_pose.translation - pole_front_direction - pole_left_direction,
                ]
            )[:, :2]

            # Check that spawn location is not occupied by extending the bounding box of the sign to include minimum
            #   distance between signs.
            if (~sim_state.current_occupancy_grid.is_occupied_world(points=points_to_check)).all():
                valid_spawn_found = True

                # Now ensure that the pole (and by extension the signs) are always facing the right direction

                if self._orient_signs_facing_travel_direction:
                    spawn_road_object = sim_state.map_query.get_road_segment(road_to_spawn_on)
                    lane_segments = [
                        sim_state.map_query.get_lane_segment(lane_id) for lane_id in spawn_road_object.lane_segments
                    ]
                    drivable_lane_segments = [
                        lane_seg for lane_seg in lane_segments if lane_seg.type == LaneSegment.LaneType.DRIVABLE
                    ]

                    # TODO: implement check to make sure we have at least two drivable lanes
                    edge_lane = (
                        drivable_lane_segments[0] if side_to_spawn_on == Side.LEFT else drivable_lane_segments[-1]
                    )

                    if edge_lane.direction == LaneSegment.Direction.BACKWARD:
                        # Rotate sign to face direction of travel
                        flipped_diff = -1 * diff
                        spawn_pose = Transformation(
                            translation=factor * diff + edge_line_to_spawn[index_to_spawn_beyond, :],
                            quaternion=Quaternion(
                                axis=(0.0, 0.0, 1.0), radians=np.arctan2(-flipped_diff[0], flipped_diff[1])
                            ),
                        )

                else:
                    # Calculate the unit vectors of the pole and ego's forward direction
                    unit_ego_front_direction = ego_forward_direction / np.linalg.norm(ego_forward_direction)
                    unit_pole_front_direction = pole_front_direction / np.linalg.norm(pole_front_direction)

                    # Calculate the dot product of the two forward direction vectors
                    angle_to_ego = np.dot(-unit_ego_front_direction, unit_pole_front_direction)

                    # If the angle is less than 90 degrees, sign is facing away from ego, and we should rotate it by
                    # 180 deg.
                    if angle_to_ego < np.cos(np.pi / 2):
                        spawn_pose = Transformation.from_transformation_matrix(
                            mat=(
                                spawn_pose.transformation_matrix
                                @ Transformation.from_euler_angles(
                                    angles=[0.0, 0.0, np.pi], order="xyz", degrees=False
                                ).transformation_matrix
                            )
                        )

            # Raise error if we exceed max retries
            elif attempts > self._max_retries:
                raise PdError(
                    "Could not find valid spawn location for sign.  "
                    "Try reducing number of signs or increasing spawn radius"
                )
            else:  # If valid spawn not found, do the loop again
                attempts += 1

        return spawn_pose

    # This method is responsible for setting the initial position of the sign pole to which we assign this behavior.
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # We use the method implemented above to find a valid location to place the sign pole
        spawn_pose = self._find_spawn_point(sim_state=sim_state, agent=agent)
        # Set the agent's pose to be the pose found above
        agent.set_pose(pose=spawn_pose.transformation_matrix)

    # This method is responsible for modifying the agent as the scenario progresses at every timestep.  If we don't want
    # the sign posts to move throughout the scene, then we can simply do nothing in this method.

    # However, in this case, we want to make the sign pole behavior compatible with an ego that change location every
    # frame, so we will update the spawn position with changing ego locations
    def update_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        # This assumes 5 start skip frames, sim capture rate of 10, and that we want to "move" 5 frames after each
        # capture frame.
        if (
            (int(sim_state.current_frame_id) % self._sim_capture_rate) == 0
            and self._single_frame_mode
            and int(sim_state.current_frame_id) > 0
        ):
            # Use the functionality implemented above to search for a new spawn position again
            spawn_pose = self._find_spawn_point(sim_state=sim_state, agent=agent)
            # Set the agent's pose to be the pose found above
            agent.set_pose(pose=spawn_pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "TrafficSignPoleBehavior":
        return TrafficSignPoleBehavior(
            random_seed=self._random_seed,
            radius=self._radius,
            forward_offset_to_place_signs=self._forward_offset_to_place_signs,
            min_distance_between_signs=self._min_distance_between_signs,
            single_frame_mode=self._single_frame_mode,
            max_retries=self._max_retries,
            orient_signs_facing_travel_direction=self._orient_signs_facing_travel_direction,
        )


class TrafficSignAttachToPoleBehavior(CustomSimulationAgentBehavior):
    """
    Behavior that spawns traffic sign poles to which traffic signs can be attached.  This behavior works by defining
        a region in which traffic sign poles can be placed, and randomly selecting points within that region to spawn
        traffic sign poles

    Args:
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        parent_pole_id: The agent ID of the traffic sign pole to which this traffic sign should be attached
        all_signs_on_pole_metadata: A list of dictionaries containing the metadata of all signs which are attached to
            the traffic sign pole defined by the parent_pole_id.  The dict should contain the fields "sign_name" and
            "height" which contain the asset name of each sign and the height (in meters) of each sign respectively.
            The information in this list of dictionaries is used to determine the vertical location on the traffic sign
            pole at which each sign should be spawned
        sign_spacing: The vertical distance (in meters) which should separate each sign on the traffic sign pole
        max_random_yaw: The maximum amount (in degrees) that the sign orientation (yaw) should be randomized around the
            forward direction.
    """

    def __init__(
        self,
        random_seed: int,
        parent_pole_id: int,
        all_signs_on_pole_metadata: List[Dict],
        sign_spacing: float,
        max_random_yaw: float,
    ):
        self._random_seed = random_seed
        self._parent_pole_id = parent_pole_id
        self._all_signs_on_pole_metadata = all_signs_on_pole_metadata
        self._sign_spacing = sign_spacing
        self._max_random_yaw = max_random_yaw

        self._random_state = random.Random(random_seed)

    # This function calculates the pose of a sign so that it sits flush on the sign's corresponding sign post
    def _find_sign_pose_on_pole(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent) -> Transformation:
        # First we extract the agent object of the pole that the sign should be placed on by iterating through the
        # agents in the sim_state and matching the agent ids
        parent_pole_object = next(agent for agent in sim_state.current_agents if agent.agent_id == self._parent_pole_id)

        # We then extract the pose of the parent pole and create a transformation object from it
        parent_pole_pose = Transformation.from_transformation_matrix(mat=parent_pole_object.pose)

        # Extract all the sign names which exist on the traffic sign pole defined by parent_pole_id
        sign_names = [sign["sign_name"] for sign in self._all_signs_on_pole_metadata]

        # Find the index in the list of all signs on the pole which corresponds to this the sign that is being placed
        # by this behavior object.  This tells us the order in which the signs exist on the pole
        sign_index = sign_names.index(agent.step_agent.asset_name)

        # Find the total vertical distance of signs + spacing that have already been placed on the traffic sign pole
        height_of_signs_already_on_pole = self._all_signs_on_pole_metadata[0]["height"] / 2
        if sign_index != 0:
            for i in range(sign_index):
                height_of_signs_already_on_pole += (
                    self._all_signs_on_pole_metadata[i - 1]["height"] + self._sign_spacing
                )

        # Find the vertical location of where this sign should be placed
        pole_height = parent_pole_object.height
        sign_vertical_offset = pole_height - height_of_signs_already_on_pole

        # Offset the sign forwards by half of the pole width, so it doesn't clip with the pole
        pole_forward_direction = parent_pole_object.width / 2 * parent_pole_pose.quaternion.rotation_matrix @ [0, 1, 0]

        # Transformation object that lets us rotate the sign around z axis
        random_yaw = Transformation.from_axis_angle(
            axis=[0, 0, 1], angle=self._random_state.uniform(-self._max_random_yaw, self._max_random_yaw), degrees=True
        )

        # Vector in forward direction (after applying random yaw)
        pole_forward_direction_random_yaw = random_yaw.rotation @ pole_forward_direction

        sign_pose_random_yaw = random_yaw @ parent_pole_pose

        # Create the pose of the sign
        sign_on_pose_translation = (
            parent_pole_pose.translation
            + coordinate_system.up * sign_vertical_offset
            + pole_forward_direction_random_yaw
        )
        sign_pose = Transformation(
            quaternion=sign_pose_random_yaw.quaternion,
            translation=sign_on_pose_translation,
        )

        # Return the sign's pose
        return sign_pose

    # This method is responsible for setting the position of the traffic sign at the start of the scenario
    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Use the method implemented above to find the pose of the traffic sign
        sign_pose = self._find_sign_pose_on_pole(sim_state=sim_state, agent=agent)

        # Set the pose of the traffic sign to be that found above
        agent.set_pose(pose=sign_pose.transformation_matrix)

    # This method is responsible for updating the state of the traffic sign every time step of the simulation.  Because
    # this behavior simply attaches traffic signs to a traffic pole, we do not need to specify whether the scenario is
    # being run in single frame mode. If the traffic poles move, this behavior will attach signs to the pole at the new
    # location.  If the traffic poles do not move, this behavior will simply return an unchanged pose.
    def update_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        # Use the method implemented above to find the pose of the traffic sign
        sign_pose = self._find_sign_pose_on_pole(sim_state=sim_state, agent=agent)

        # Set the pose of the traffic sign to be that found above
        agent.set_pose(pose=sign_pose.transformation_matrix)

    # The clone method returns a copy of the Custom Behavior object and is required under the hood by Data Lab
    def clone(self) -> "TrafficSignAttachToPoleBehavior":
        return TrafficSignAttachToPoleBehavior(
            random_seed=self._random_seed,
            parent_pole_id=self._parent_pole_id,
            all_signs_on_pole_metadata=self._all_signs_on_pole_metadata,
            sign_spacing=self._sign_spacing,
            max_random_yaw=self._max_random_yaw,
        )
