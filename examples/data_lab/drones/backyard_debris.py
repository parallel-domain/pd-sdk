import csv
import dataclasses
import logging
import random
import warnings
from typing import Callable, List, Optional

import cv2
import numpy as np
from pd.core import PdError
from pd.data_lab.context import load_map, setup_datalab
from pd.data_lab.generators.custom_simulation_agent import CustomSimulationAgent, CustomSimulationAgents
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance
from pd.internal.assets.asset_registry import ObjAssets, UtilAssetCategories
from pd.internal.proto.umd.generated.python import UMD_pb2

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.map import Area, MapQuery
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL
from paralleldomain.utilities.fsio import read_json_str, write_png
from paralleldomain.utilities.geometry import random_point_within_2d_polygon
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

setup_datalab("v2.4.1-beta")
LOCATION = "SC_W8thAndOrchard"


def get_asset_list() -> List[str]:
    # query specific assets from PD's asset registry and use as csv input for Debris generator
    asset_objs = (
        ObjAssets.select(ObjAssets.name)
        .join(UtilAssetCategories)
        .where((UtilAssetCategories.name == "prop") & ((ObjAssets.name % "*bbq*") | (ObjAssets.name % "*chair*")))
    )
    asset_names = [o.name for o in asset_objs]
    return asset_names


def get_random_area_subtype_object(
    area_type: UMD_pb2.Area.AreaType, sub_type: str, random_seed: int
) -> Optional[UMD_pb2.Area]:
    random_state = random.Random(random_seed)
    area_ids = [
        area_id
        for area_id, area in umd_map.areas.items()
        if area.type is area_type and read_json_str(area.user_data)["sub_type"] == sub_type
    ]
    if len(area_ids) == 0:
        return None
    area_id = random_state.choice(area_ids)
    return umd_map.areas.get(area_id)


class EgoDroneStraightLineBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        start_pose: Transformation,
        target_pose: Transformation,
        flight_time: float,
    ):
        super().__init__()
        self._initial_pose: Transformation = start_pose
        self._target_pose: Transformation = target_pose
        self._flight_time: float = flight_time
        self._start_time: Optional[float] = None

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        agent.set_pose(pose=self._initial_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        current_time = sim_state.sim_time

        if self._start_time is None:
            self._start_time = current_time  # set first frame as start time even if not exactly 0.0 seconds

        flight_completion = current_time / (self._flight_time + self._start_time)

        interpolated_pose = Transformation.interpolate(
            tf0=self._initial_pose, tf1=self._target_pose, factor=flight_completion
        )

        logger.info(f"Using interpolated pose: {interpolated_pose}")
        agent.set_pose(pose=interpolated_pose.transformation_matrix)

    def clone(self) -> "EgoDroneStraightLineBehaviour":
        return EgoDroneStraightLineBehaviour(
            start_pose=self._initial_pose,
            target_pose=self._target_pose,
            flight_time=self._flight_time,
        )


class StaticAssetBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self, location_xyz: np.ndarray, polygon: np.ndarray, apply_random_yaw: bool = True, max_attempts: int = 10
    ):
        super().__init__()
        self._location_xyz: np.ndarray = location_xyz
        self._polygon: np.ndarray = polygon
        self._apply_random_yaw: bool = apply_random_yaw
        self._max_attempts: int = max_attempts
        self._pose: Optional[Transformation] = None
        self._found_valid_placement: bool = False

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        """
        If the original requested spawn location fails due to a collision, try another random point within
        the same polygon, up to max_attempts tries. If no collision-free point is found, it will place the asset
        anyway.
        """

        random_state = random.Random(random_seed + agent.agent_id)

        asset_width, asset_length, asset_height = CustomSimulationAgents.get_asset_size(
            asset_name=agent.step_agent.asset_name
        )

        attempts = 0
        while attempts < self._max_attempts and not self._found_valid_placement:
            self._pose = Transformation(translation=self._location_xyz)

            # Apply random yaw rotation
            if self._apply_random_yaw:
                random_rotation_radians = random_state.uniform(a=0, b=2 * np.pi)
                random_rotation = Transformation.from_euler_angles(
                    angles=[0, 0, random_rotation_radians], order="xyz", degrees=False
                )
                self._pose = self._pose @ random_rotation

            asset_box = BoundingBox3DGeometry(
                pose=self._pose @ SIM_TO_INTERNAL.inverse,
                width=asset_width,
                height=asset_height,
                length=asset_length,
            )
            vertices = asset_box.vertices

            # Can't just check the four corners, need to check to see if points inside object are occupied
            interior_points_to_check = random_point_within_2d_polygon(
                edge_2d=vertices[:, :2],
                random_seed=scenario.random_seed,
                num_points=100,
            )

            total_points_to_check = np.vstack((vertices[:, :2], interior_points_to_check))

            if not np.any(sim_state.current_occupancy_grid.is_occupied_world(points=total_points_to_check)):
                self._found_valid_placement = True
                continue

            # Seeding here is a bit hacky
            self._location_xyz = np.append(
                random_point_within_2d_polygon(
                    edge_2d=vertices[:, :2],
                    random_seed=random_seed + agent.agent_id + attempts,
                    num_points=1,
                ),
                0,
            )

            attempts += 1

        if not self._found_valid_placement:
            warnings.warn(
                f"Could not find a collision-free placement for asset {agent.step_agent.asset_name}. Placing with "
                f"possible collision "
            )

        agent.set_pose(pose=self._pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "StaticAssetBehaviour":
        return StaticAssetBehaviour(
            location_xyz=self._location_xyz, apply_random_yaw=self._apply_random_yaw, polygon=self._polygon
        )


@dataclasses.dataclass
class AreaDebrisGenerator(data_lab.CustomAtomicGenerator):
    area_id: int
    number_of_assets_to_place: int
    apply_random_yaw: bool = True

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []
        random_state = random.Random(random_seed)

        # Get list of assets to choose randomly from
        asset_list = get_asset_list()

        # Here we are finding some random points to spawn assets inside the yard boundary
        yard_boundary = umd_map.edges[int(umd_map.areas[self.area_id].edges[0])].as_polyline().to_numpy()

        debris_points = random_point_within_2d_polygon(
            edge_2d=yard_boundary[:, :2],
            random_seed=scenario.random_seed + 1,
            num_points=self.number_of_assets_to_place,
        )

        for i in range(self.number_of_assets_to_place):
            asset_to_place = random_state.choice(asset_list)

            agent = data_lab.CustomSimulationAgents.create_object(
                asset_name=asset_to_place, lock_to_ground=True
            ).set_behaviour(
                StaticAssetBehaviour(
                    location_xyz=np.array([debris_points[i, 0], debris_points[i, 1], 0]), polygon=yard_boundary[:, :2]
                )
            )
            agents.append(agent)

        return agents

    def clone(self):
        return AreaDebrisGenerator(
            area_id=self.area_id,
            number_of_assets_to_place=self.number_of_assets_to_place,
            apply_random_yaw=self.apply_random_yaw,
        )


sensor_rig = data_lab.SensorRig().add_camera(
    name="Front",
    width=768,
    height=768,
    field_of_view_degrees=70,
    pose=Transformation.from_euler_angles(
        angles=[-90, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 0.0]
    ),
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.wetness.set_uniform_distribution(min_value=0, max_value=0)

# Select an environment
location = data_lab.Location(name=LOCATION)
scenario.set_location(location)

# Load map to find a random yard spawn point and its XYZ coordinates
umd_map = load_map(location)
map_query = MapQuery(umd_map)

# Ensure we are getting a backyard spawn area
area = get_random_area_subtype_object(Area.AreaType.YARD, sub_type="yard_back_direct", random_seed=scenario.random_seed)
# If there are no yards that meet criteria
if area is None:
    raise PdError("Failed to find Yard location to spawn. Please try another map")

area_id = area.id
# Get a random point within yard to use as starting location for drone
start_pose = map_query.get_random_area_location(
    area_type=Area.AreaType.YARD, random_seed=scenario.random_seed, area_id=area_id
)

end_pose = Transformation.from_transformation_matrix(mat=start_pose.transformation_matrix)

# map query gives us ground position, but we want our Drone to end 0.25 above ground
end_pose.translation[2] += 0.25
start_pose.translation[2] += 20

# Place ourselves in the world through a custom simulation agent. Don't use an asset so we don't see anything flying
# attach our EgoDroneBehavior from above
scenario.add_ego(
    data_lab.CustomSimulationAgents.create_ego_vehicle(
        sensor_rig=sensor_rig,
        asset_name="",
        lock_to_ground=False,
    ).set_behaviour(EgoDroneStraightLineBehaviour(start_pose=start_pose, target_pose=end_pose, flight_time=3.0))
)

# Place objects in backyard
area_gen = AreaDebrisGenerator(area_id=area_id, number_of_assets_to_place=20)
scenario.add_objects(area_gen)

data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
