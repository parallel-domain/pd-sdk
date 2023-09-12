import random
from typing import Optional, Callable, List, Dict

import numpy as np
from pd.core import PdError
from pyquaternion import Quaternion

from paralleldomain.data_lab import (
    ExtendedSimState,
    CustomSimulationAgent,
    CustomSimulationAgentBehaviour,
    coordinate_system,
)
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.utilities.geometry import random_point_within_2d_polygon
from paralleldomain.utilities.transformation import Transformation


class TrafficSignPoleBehavior(CustomSimulationAgentBehaviour):
    """
    Behavior that spawns traffic sign poles to which traffic signs can be attached.  This behavior works by defining
        a region in which traffic sign poles can be placed, and randomly selecting points within that region to spawn
        traffic sign poles

    Args:
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        radius: The radius of the region in which traffic signs should be spawned
        single_frame_mode: Flag to indicate whether this generator is being used to configure a single frame
            dataset.  In single frame datasets, the location of the vehicles and the signage changes between each
            rendered frame
        forward_offset_to_place_signs:  The distance (in meters) in front of the ego vehicle on which the region in
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
        forward_offset_to_place_signs: float = 15,
        min_distance_between_signs: float = 2.5,
        max_retries: int = 1000,
    ):
        self._random_seed = random_seed
        self._radius = radius
        self._sim_capture_rate = 10  # Enforced for single frames, not needed for temporal scenes
        self._forward_offset_to_place_signs = forward_offset_to_place_signs
        self._min_distance_between_signs = min_distance_between_signs
        self._single_frame_mode = single_frame_mode
        self._max_retries = max_retries

        self._random_state = random.Random(self._random_seed)

    # Function which handle the logic of selecting where to place the traffic sign pole
    def _find_spawn_point(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent) -> Transformation:
        # Define a vector which denotes how far in front of the ego the center of the valid spawn region should be
        ego_forward_direction = (
            self._forward_offset_to_place_signs * sim_state.ego_pose.quaternion.rotation_matrix @ np.array([0, 1, 0])
        )

        # Center of the valid spawn region and create the Transformation Object
        search_center_location = sim_state.ego_pose.translation + ego_forward_direction
        search_center_pose = Transformation.from_euler_angles(
            angles=sim_state.ego_pose.as_euler_angles(order="xyz", degrees=False),
            order="xyz",
            degrees=False,
            translation=search_center_location,
        )

        # Find all sidewalk lanes near the ego within the defined radius
        sidewalks_near = [
            lane
            for lane in sim_state.map_query.get_lane_segments_near(
                pose=search_center_pose, radius=self._radius, method="overlap"
            )
            if lane.type is LaneSegment.LaneType.SIDEWALK
        ]

        # Perform collision checks in a loop
        valid_spawn_found = False
        attempts = 0
        while not valid_spawn_found:
            sidewalk_to_spawn_on = self._random_state.choice(
                sidewalks_near
            )  # Randomly choose one of the sidewalk lanes

            # Find the left and right edges of the sidewalk lane
            sidewalk_left_edge = sim_state.map.edges[sidewalk_to_spawn_on.left_edge].as_polyline().to_numpy()
            sidewalk_right_edge = sim_state.map.edges[sidewalk_to_spawn_on.right_edge].as_polyline().to_numpy()

            # Find the minimum distance between the ego and the left/right lane respectively
            left_edge_distance = np.min(np.linalg.norm(sidewalk_left_edge - sim_state.ego_pose.translation, axis=1))
            right_edge_distance = np.min(np.linalg.norm(sidewalk_right_edge - sim_state.ego_pose.translation, axis=1))

            # Choose the edge which is closest to the ego vehicle
            edge_line_to_spawn = sidewalk_left_edge if left_edge_distance < right_edge_distance else sidewalk_right_edge

            # Randomly choose where along the line to space
            index_to_spawn_beyond = self._random_state.randint(0, edge_line_to_spawn.shape[0] - 2)
            factor = self._random_state.uniform(0.0, 1.0)

            # Calculate the diff between the preceding and succeeding point on the line
            diff = edge_line_to_spawn[index_to_spawn_beyond + 1, :] - edge_line_to_spawn[index_to_spawn_beyond]

            # Create a pose containing the potential spawn location of the pole
            spawn_pose = Transformation(
                translation=factor * diff + edge_line_to_spawn[index_to_spawn_beyond, :],
                quaternion=Quaternion(axis=(0.0, 0.0, 1.0), radians=np.arctan2(-diff[0], diff[1])),
            )

            # Find the left and forward directions from the pole (incorporating the min_distance_between_signs)
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

            # Create a box and fill it with points for the occupancy check
            points_to_check = np.array(
                [
                    spawn_pose.translation + pole_front_direction + pole_left_direction,
                    spawn_pose.translation + pole_front_direction - pole_left_direction,
                    spawn_pose.translation - pole_front_direction + pole_left_direction,
                    spawn_pose.translation - pole_front_direction - pole_left_direction,
                ]
            )[:, :2]
            filled_points = random_point_within_2d_polygon(
                edge_2d=points_to_check, random_seed=self._random_seed, num_points=25
            )
            points_to_check = np.vstack((filled_points, points_to_check))

            # Check that the spawn pose is within the specified radius of the valid spawn region and the location is
            # not occupied
            if (
                np.linalg.norm(spawn_pose.translation - search_center_pose.translation) <= self._radius
                and (~sim_state.current_occupancy_grid.is_occupied_world(points=points_to_check)).all()
            ):
                valid_spawn_found = True

                # Now ensure that the pole (and by extension the signs) are always facing the ego

                # Calculate the unit vector of the pole and ego's forward direction
                unit_ego_front_direction = ego_forward_direction / np.linalg.norm(ego_forward_direction)
                unit_pole_front_direction = pole_front_direction / np.linalg.norm(pole_front_direction)

                # Calculate the dot product of the two forward direction vectors
                angle_to_ego = np.dot(-unit_ego_front_direction, unit_pole_front_direction)

                # If the angle is less than 90 degrees, sign is facing away from ego, and we should rotate it by 180 deg
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

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        spawn_pose = self._find_spawn_point(sim_state=sim_state, agent=agent)

        agent.set_pose(pose=spawn_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        if (
            int(sim_state.current_frame_id) % self._sim_capture_rate
        ) - 5 == 1 and self._single_frame_mode:  # Change location right after capture for antialiasing
            spawn_pose = self._find_spawn_point(sim_state=sim_state, agent=agent)

            agent.set_pose(pose=spawn_pose.transformation_matrix)

    def clone(self) -> "TrafficSignPoleBehavior":
        return TrafficSignPoleBehavior(
            random_seed=self._random_seed,
            radius=self._radius,
            forward_offset_to_place_signs=self._forward_offset_to_place_signs,
            min_distance_between_signs=self._min_distance_between_signs,
            single_frame_mode=self._single_frame_mode,
            max_retries=self._max_retries,
        )


class TrafficSignAttachToPoleBehavior(CustomSimulationAgentBehaviour):
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
    """

    def __init__(
        self,
        random_seed: int,
        parent_pole_id: int,
        all_signs_on_pole_metadata: List[Dict],
        sign_spacing: float,
    ):
        self._random_seed = random_seed
        self._parent_pole_id = parent_pole_id
        self._all_signs_on_pole_metadata = all_signs_on_pole_metadata
        self._sign_spacing = sign_spacing

    # Function which determines the spawn location of each sign
    def _find_sign_pose_on_pole(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent) -> Transformation:
        # Find the parent pole object from the sim state
        parent_pole_object = next(agent for agent in sim_state.current_agents if agent.agent_id == self._parent_pole_id)
        parent_pole_pose = Transformation.from_transformation_matrix(mat=parent_pole_object.pose)

        # Pull all the sign names which exist on the traffic sign pole defined by parent_pole_id
        sign_names = [sign["sign_name"] for sign in self._all_signs_on_pole_metadata]

        # Find the index in the list of all signs on the pole which corresponds to this the sign that is being placed
        # by this behavior object
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

        # Create the pose of the sign
        sign_on_pose_translation = (
            parent_pole_pose.translation + coordinate_system.up * sign_vertical_offset + pole_forward_direction
        )
        sign_pose = Transformation.from_euler_angles(
            angles=parent_pole_pose.as_euler_angles(
                order="xyz",
                degrees=False,
            ),
            order="xyz",
            translation=sign_on_pose_translation,
            degrees=False,
        )

        return sign_pose

    def set_initial_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        sign_pose = self._find_sign_pose_on_pole(sim_state=sim_state, agent=agent)
        agent.set_pose(pose=sign_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: ExtendedSimState,
        agent: CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        # Note this will work in both single frame and temporal mode because movement is goverened by the pole behavior
        # only
        sign_pose = self._find_sign_pose_on_pole(sim_state=sim_state, agent=agent)
        agent.set_pose(pose=sign_pose.transformation_matrix)

    def clone(self) -> "TrafficSignAttachToPoleBehavior":
        return TrafficSignAttachToPoleBehavior(
            random_seed=self._random_seed,
            parent_pole_id=self._parent_pole_id,
            all_signs_on_pole_metadata=self._all_signs_on_pole_metadata,
            sign_spacing=self._sign_spacing,
        )
