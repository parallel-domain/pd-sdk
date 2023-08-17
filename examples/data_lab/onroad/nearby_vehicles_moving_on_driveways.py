import logging
import random

import numpy as np
from pd.core.errors import PdError
from pd.data_lab import Scenario
from pd.data_lab.context import load_map, setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

from paralleldomain.data_lab import Location, SensorConfig, SensorRig, TimeOfDays, preview_scenario
from paralleldomain.data_lab.config.map import LaneSegment, MapQuery, RoadSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.types import Float3
from paralleldomain.data_lab.generators.driveway import DrivewayCreepGenerator
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logger = logging.getLogger("pd.state.serialize")
logger.setLevel(logging.CRITICAL)

setup_datalab("v2.4.0-beta")

sensor_rig = SensorRig(
    sensor_configs=[
        SensorConfig(
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
scenario = Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.0)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_constant_value(0.0)
scenario.environment.wetness.set_constant_value(0.0)

# Select an environment
location = Location(name="A2_Kerrytown")
scenario.set_location(location=location)

umd_map = load_map(location=location)
map_query = MapQuery(umd_map)

# Set up scene and driveway parameters
radius_to_driveways = 30
minimum_driveway_length = 6.0

# Search for a valid spawn position
valid_spawn_found = False
retries = 0
while not valid_spawn_found:
    driveway_point = map_query.get_random_road_type_object(
        road_type=RoadSegment.RoadType.DRIVEWAY, random_seed=scenario.random_seed
    )

    if driveway_point is None:
        raise PdError("No available driveways exist on map.  Select another map.")

    driveway_lane = umd_map.lane_segments[int(driveway_point.lane_segments[0])]
    driveway_reference_line = map_query.edges[int(driveway_lane.reference_line)].as_polyline().to_numpy()
    driveway_reference_point = driveway_reference_line[0]

    # Check against minimum driveway length
    if np.linalg.norm(driveway_reference_line[-1] - driveway_reference_line[0]) < 1.15 * minimum_driveway_length:
        continue

    bounds = BoundingBox2DBaseGeometry(
        x=driveway_reference_point[0] - radius_to_driveways,
        y=driveway_reference_point[1] - radius_to_driveways,
        width=2 * radius_to_driveways,
        height=2 * radius_to_driveways,
    )

    lanes_near_spawn_point = map_query.get_lane_segments_within_bounds(bounds=bounds, method="overlap")

    spawn_point_segment = next((ls for ls in lanes_near_spawn_point if ls.type is LaneSegment.LaneType.DRIVABLE), None)

    if spawn_point_segment is None:
        logger.info(f"Failed to find valid spawn point at seed {scenario.random_seed} - incrementing and retrying")
        scenario.random_seed += 1
    else:
        # We find the point on the reference line that is within the bounds of the driveway
        spawn_point_line = map_query.edges[int(spawn_point_segment.reference_line)].as_polyline().to_numpy()
        distances_to_driveway = np.linalg.norm(spawn_point_line - driveway_reference_point, axis=1)

        spawn_point = spawn_point_line[np.argmin(distances_to_driveway)]

        valid_spawn_found = True

    if retries >= 1000:
        raise PdError(
            "Failed to find valid spawn location within max_retries set.  Adjust max_retries or spawn conditions"
        )

    retries += 1

scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            absolute_position_request=AbsolutePositionRequest(
                position=Float3(x=spawn_point[0], y=spawn_point[1]), resolve_z=True
            )
        ),
        vehicle_spawn_data=VehicleSpawnData(vehicle_peripheral=VehiclePeripheral(disable_occupants=True)),
    ),
)

scenario.add_agents(
    generator=DrivewayCreepGenerator(
        behavior_duration=2.0,
        number_of_vehicles=3,
        driveway_entry_probability=0.5,
        radius=radius_to_driveways,
        min_driveway_length=minimum_driveway_length,
    )
)

scenario.add_agents(
    generator=TrafficGeneratorParameters(
        spawn_probability=0.8,
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
            )
        ),
    )
)

preview_scenario(
    scenario=scenario,
    frames_per_scene=20,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
