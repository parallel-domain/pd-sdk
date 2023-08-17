import dataclasses
import logging
import random
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
from pd.assets import DataSign, InfoSegmentation, ObjAssets, ObjCountries, UtilSegmentationCategoriesPanoptic
from pd.data_lab.config.distribution import CenterSpreadConfig, Distribution, EnumDistribution
from pd.data_lab.context import load_map, setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance
from pd.sim import Raycast

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.map import Edge, LaneSegment, MapQuery, RoadSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.4.0-beta")

SIGN_POST_LIST = [
    "post_round_metal_0365h_06r",
    "post_sign_0288h02r",
    "crosswalkpost_01_small",
    "post_round_metal_0400h_08r",
]
INVALID_ROAD_TYPES = [
    RoadSegment.RoadType.UNCLASSIFIED,
    RoadSegment.RoadType.DRIVEWAY,
    RoadSegment.RoadType.MOTORWAY,
    RoadSegment.RoadType.PARKING_AISLE,
    RoadSegment.RoadType.DRIVEWAY_PARKING_ENTRY,
]

LOCATION = "SF_6thAndMission_medium"


class EdgeType(Enum):
    LEFT = 1
    RIGHT = 2


def get_all_country_signs_with_dimensions(country: str) -> list:
    query = (
        InfoSegmentation.select(
            InfoSegmentation.name.alias("sign_name"),
            ObjCountries.name.alias("country"),
            ObjAssets.width,
            ObjAssets.length,
            ObjAssets.height,
        )
        .join(
            UtilSegmentationCategoriesPanoptic,
            on=(InfoSegmentation.panoptic_id == UtilSegmentationCategoriesPanoptic.id),
        )
        .join(DataSign, on=(DataSign.asset_id == ObjAssets.id))
        .join(ObjCountries, on=(ObjCountries.id == DataSign.country_id))
        .join(ObjAssets, on=(ObjAssets.id == InfoSegmentation.asset_id))
        .where(
            UtilSegmentationCategoriesPanoptic.name == "TrafficSign",
            ObjCountries.name == country,
        )
    ).dicts()

    sign_list = [sign for sign in query]
    return sign_list


def get_all_sign_posts_with_dimensions() -> list:
    query = (
        InfoSegmentation.select(InfoSegmentation.name, ObjAssets.width, ObjAssets.height, ObjAssets.length)
        .join(ObjAssets, on=(InfoSegmentation.asset_id == ObjAssets.id))
        .where(ObjAssets.name << SIGN_POST_LIST)
    ).dicts()

    post_list = [post for post in query]

    return post_list


SIGN_POSTS = get_all_sign_posts_with_dimensions()


def get_valid_lane_ids(lane_segment_ids: List[int]):
    valid_lane_ids = []
    for lane_id in lane_segment_ids:
        lane = map_query.get_lane_segment(lane_id)
        if lane.type in [LaneSegment.LaneType.PARKING, LaneSegment.LaneType.DRIVABLE]:
            valid_lane_ids.append(lane_id)
    return valid_lane_ids


def get_offset_point_3d(
    point_a: np.ndarray,
    point_b: np.ndarray,
    edge_type: Enum,
    curb_offset: float,
    walk_distance: float,
) -> np.ndarray:
    v_a_b = point_b - point_a
    v_hat_a_b = v_a_b / np.linalg.norm(v_a_b)
    mid = point_a + walk_distance * v_hat_a_b

    if edge_type == EdgeType.LEFT:
        # Counterclockwise perpendicular vector
        n = np.array([-v_a_b[1], v_a_b[0], 0])
    else:
        # Clockwise perpendicular vector
        n = np.array([v_a_b[1], -v_a_b[0], 0])
    n_norm = n / np.linalg.norm(n)
    offset_point_3d = mid + curb_offset * n_norm
    return offset_point_3d


sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig(
            display_name="Front",
            camera_intrinsic=CameraIntrinsic(
                width=1920,
                height=1080,
                fov=70.0,
            ),
            sensor_extrinsic=SensorExtrinsic(
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                x=0.0,
                y=0.0,
                z=2.0,
            ),
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_uniform_distribution(0, 0.3)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0, max_value=0.1)
scenario.environment.wetness.set_uniform_distribution(min_value=0.0, max_value=0.2)

location = data_lab.Location(name=LOCATION)

# Select an environment
scenario.set_location(location)

# Initialize umd
umd_map = load_map(location)
map_query = MapQuery(umd_map)


class TrafficSignBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        location_xyz: np.ndarray,
        height_on_pole: float,
        pole_width: float,
        v_hat_sign: np.ndarray,
    ):
        super().__init__()
        self._location_xyz = location_xyz
        self._v_hat_sign = v_hat_sign  # This represents the desired surface normal (for now I only use the z rotation)
        self._height_on_pole = height_on_pole
        self._pole_width = pole_width

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Calculate offset transformation to orient sign properly on pole
        sign_position_above_ground_xyz = self._location_xyz + data_lab.coordinate_system.up * self._height_on_pole
        offset_m = self._pole_width / 2
        final_sign_position_world_coords = sign_position_above_ground_xyz + (self._v_hat_sign * offset_m)

        # Get rotation around Z from desired surface normal vector
        z_rot_radians = np.arctan2(self._v_hat_sign[1], self._v_hat_sign[0]) + (np.pi / 2)

        sign_rotation_trans = Transformation.from_euler_angles(
            angles=[0, 0, z_rot_radians + np.pi], degrees=False, order="xyz"
        )
        sign_quat = sign_rotation_trans.quaternion

        sign_pose = Transformation(quaternion=sign_quat, translation=final_sign_position_world_coords)

        agent.set_pose(pose=sign_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "TrafficSignBehaviour":
        return TrafficSignBehaviour(
            location_xyz=self._location_xyz,
            v_hat_sign=self._v_hat_sign,
            pole_width=self._pole_width,
            height_on_pole=self._height_on_pole,
        )


class TrafficPoleBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(self, pose: Transformation):
        super().__init__()
        self.pose = pose

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        ray_start = self.pose.translation + data_lab.coordinate_system.up * 10

        result = raycast(  # send one ray, but possible to send multiple to cover wider area
            [
                Raycast(
                    origin=tuple(ray_start),
                    direction=tuple(data_lab.coordinate_system.down),
                    max_distance=20,
                )
            ]
        )
        final_pose = Transformation(quaternion=self.pose.quaternion, translation=result[0][0].position)
        agent.set_pose(pose=final_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "TrafficPoleBehaviour":
        return TrafficPoleBehaviour(pose=self.pose)


@dataclasses.dataclass
class SignGenerator(data_lab.CustomAtomicGenerator):
    country_weights: Dict = dataclasses.field(default_factory=dict)
    density: Distribution = Distribution.create(value=0)
    sign_spacing: Distribution = Distribution.create(value=(0.08, 0.12))
    min_sign_distance_to_ground: int = 0.25
    min_pole_signs: int = 1
    max_pole_signs: int = 4
    curb_offset = Distribution.create(value=(0.35, 0.6))

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        # Grab all roads within (approximately) 200m
        road_segments = self.get_valid_road_segments_near_ego(state.ego_pose, radius=200)

        # Query the database to grab all signs for each specified country, along with dimensions
        country_signs = {
            country: get_all_country_signs_with_dimensions(country) for country in self.country_weights.keys()
        }

        agents = list()
        for seg in road_segments:
            # Valid lane ids are lanes that are not parking spaces.
            # We can't use parking space edges as left or right edges

            valid_lane_ids = get_valid_lane_ids(seg.lane_segments)

            # Fix this to perform sign placing for all roads
            left_lane_id = valid_lane_ids[0]
            right_lane_id = valid_lane_ids[-1]

            left_lane = map_query.get_lane_segment(int(left_lane_id))
            right_lane = map_query.get_lane_segment(int(right_lane_id))

            # This is needed if placing signs so that they face the direction of travel
            left_lane_forward_direction = True if left_lane.direction == LaneSegment.Direction.FORWARD else False
            right_lane_forward_direction = True if right_lane.direction == LaneSegment.Direction.FORWARD else False

            # This logic ensures we are finding the correct (outside) edges of the road.
            if left_lane_forward_direction:
                left_lane_left_edge_id = left_lane.left_edge
            else:
                left_lane_left_edge_id = left_lane.right_edge

            if right_lane_forward_direction:
                right_lane_right_edge_id = right_lane.right_edge
            else:
                right_lane_right_edge_id = right_lane.left_edge

            left_lane_left_edge = map_query.get_edge(int(left_lane_left_edge_id))
            right_lane_right_edge = map_query.get_edge(int(right_lane_right_edge_id))

            agents = agents + self.place_signs_along_edge(
                e=left_lane_left_edge,
                edge_type=EdgeType.LEFT,
                is_forward_lane=left_lane_forward_direction,
                country_signs=country_signs,
            )
            agents = agents + self.place_signs_along_edge(
                e=right_lane_right_edge,
                edge_type=EdgeType.RIGHT,
                is_forward_lane=right_lane_forward_direction,
                country_signs=country_signs,
            )

        return agents

    def get_valid_road_segments_near_ego(
        self, ego_pose: Transformation, radius: Optional[int] = 200
    ) -> List[LaneSegment]:
        bounds = BoundingBox2DBaseGeometry(
            x=ego_pose.translation[0] - (radius / 2),
            y=ego_pose.translation[1] - (radius / 2),
            width=radius,
            height=radius,
        )
        road_segments = map_query.get_road_segments_within_bounds(bounds, method="overlap")

        # Here we exclude some road types we won't want to place signs beside (like driveways)
        road_segments = [seg for seg in list(road_segments) if seg.type not in INVALID_ROAD_TYPES]
        return road_segments

    def place_signs_along_edge(
        self, e: Edge, edge_type: Enum, is_forward_lane: bool, country_signs: Dict
    ) -> List[data_lab.CustomSimulationAgent]:
        sign_agents = []

        # First random walk length along edge
        random_walk_distance = self.density.sample(random_seed=np.random.uniform(low=0, high=999999))

        # This will be for running count of distance left to travel on current random walk
        remaining_walk_distance = random_walk_distance

        for i in range(len(e.points) - 1):
            point_a = e.points[i]
            point_b = e.points[i + 1]

            # Get unit vector in same direction as line a -> b of edge
            point_a_3d = np.array([point_a.x, point_a.y, point_a.z])
            point_b_3d = np.array([point_b.x, point_b.y, point_b.z])
            v_a_b = point_b_3d - point_a_3d
            len_a_b = np.linalg.norm(v_a_b)
            v_hat_a_b = v_a_b / len_a_b

            # Needed to determine which way sign should face
            offset_direction = (-1) if is_forward_lane else 1
            v_hat_sign = offset_direction * v_hat_a_b

            # This tracks how far we have already travelled on line a -> b.
            already_walked_in_segment = 0

            while (already_walked_in_segment + remaining_walk_distance) < len_a_b:
                # Get point 'away' from the road, so that pole is placed on sidewalk reasonably
                offset_point_3d = get_offset_point_3d(
                    point_a_3d,
                    point_b_3d,
                    edge_type=edge_type,
                    curb_offset=self.curb_offset.sample(random_seed=np.random.uniform(low=0, high=999999)),
                    walk_distance=already_walked_in_segment + remaining_walk_distance,
                )
                offset_point_trans = Transformation(translation=offset_point_3d)

                pole_to_place = random.choice(SIGN_POSTS)
                pole_name = pole_to_place["name"]

                # Place pole
                pole_agent = data_lab.CustomSimulationAgents.create_object(asset_name=pole_name).set_behaviour(
                    TrafficPoleBehaviour(pose=offset_point_trans)
                )

                sign_agents.append(pole_agent)

                country_to_place_sign_from = random.choices(
                    list(self.country_weights.keys()),
                    weights=tuple(self.country_weights.values()),
                    k=1,
                )

                # Place random number of signs on pole, but only if there is enough room
                highest_unoccupied_point_on_pole = pole_to_place["height"]
                for j in range(np.random.randint(self.min_pole_signs, self.max_pole_signs + 1)):
                    # Grab random sign from whatever country we randomly selected
                    sign_to_place = random.choice(country_signs[country_to_place_sign_from[0]])
                    sign_name = sign_to_place["sign_name"]

                    current_sign_spacing = self.sign_spacing.sample(random_seed=np.random.uniform(low=0, high=999999))
                    # Ensuring we have enough space to place sign
                    if (
                        sign_to_place["height"] + current_sign_spacing
                        < highest_unoccupied_point_on_pole - self.min_sign_distance_to_ground
                    ):
                        # Place sign. Exact placement of sign on pole is handled inside traffic sign behaviour class.
                        sign_height_on_pole = (
                            highest_unoccupied_point_on_pole - (sign_to_place["height"] / 2) - current_sign_spacing
                        )
                        agent = data_lab.CustomSimulationAgents.create_object(asset_name=sign_name).set_behaviour(
                            TrafficSignBehaviour(
                                location_xyz=offset_point_3d,
                                height_on_pole=sign_height_on_pole,
                                pole_width=pole_to_place["width"],
                                v_hat_sign=v_hat_sign,
                            )
                        )

                        highest_unoccupied_point_on_pole = (
                            highest_unoccupied_point_on_pole - sign_to_place["height"] - current_sign_spacing
                        )
                        sign_agents.append(agent)

                already_walked_in_segment = already_walked_in_segment + remaining_walk_distance
                random_walk_distance = self.density.sample(random_seed=np.random.uniform(low=0, high=999999))
                remaining_walk_distance = random_walk_distance

            remaining_walk_distance = remaining_walk_distance - (len_a_b - already_walked_in_segment)

        return sign_agents

    def clone(self):
        return SignGenerator(
            country_weights=self.country_weights,
            density=self.density.clone(),
            sign_spacing=self.sign_spacing.clone(),
        )


scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            lane_spawn_policy=LaneSpawnPolicy(
                lane_type=EnumDistribution(
                    probabilities={"Drivable": 1.0},
                )
            ),
        ),
    )
)

scenario.add_agents(
    generator=TrafficGeneratorParameters(
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=50.0,
            )
        ),
    )
)

scenario.add_agents(
    generator=ParkedVehicleGeneratorParameters(
        spawn_probability=CenterSpreadConfig(center=0.4),
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=50.0,
            )
        ),
    )
)

# Custom obstacle generator
sign_generator = SignGenerator()
sign_generator.country_weights = {"Turkey": 0.5, "Portugal": 0.5}
sign_generator.density.set_uniform_distribution(min_value=1, max_value=16)
scenario.add_agents(sign_generator)

data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
