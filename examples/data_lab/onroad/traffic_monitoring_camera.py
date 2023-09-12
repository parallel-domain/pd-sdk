import logging

import numpy as np
from pd.data_lab import Scenario
from pd.data_lab.config.distribution import CenterSpreadConfig, ContinousUniformDistribution
from pd.data_lab.context import load_map, setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

from paralleldomain.data_lab import (
    CustomSimulationAgents,
    Location,
    SensorConfig,
    SensorRig,
    TimeOfDays,
    preview_scenario,
    DEFAULT_DATA_LAB_VERSION,
)
from paralleldomain.data_lab.config.map import MapQuery, RoadSegment, Side
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.types import Float3
from paralleldomain.data_lab.generators.behavior import LookAtPointBehavior, VehicleBehavior
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import AgentSpawnData, VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.data_lab.generators.vehicle import VehicleGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
setup_datalab(DEFAULT_DATA_LAB_VERSION)


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
                z=0.0,
            ),
        )
    ]
)

# Create scenario
scenario = Scenario(sensor_rig=sensor_rig)
scenario.random_seed = 0  # random.randint(1, 100000)  # Fix the seed to make scenario generation deterministic


# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(TimeOfDays.Dusk, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
location = Location(name="MV_280AndPageMill")
scenario.set_location(location=location)

umd_map = load_map(location=location)
map_query = MapQuery(umd_map)

# Set up scene parameters (all distances in meters)
min_path_length = 250
camera_height = 10
distance_to_traffic = 125

# Get the spawn location for the star vehicle that assets will be placed relative to
spawn_lane = map_query.get_random_lane_object_from_road_type(
    road_type=RoadSegment.RoadType.MOTORWAY, random_seed=scenario.random_seed, min_path_length=min_path_length
)
spawn_point = umd_map.edges[spawn_lane.reference_line].as_polyline().to_numpy()[0]

# Find the point ahead of the star vehicle where the traffic jam will be located
star_vehicle_path = map_query.get_connected_lane_points(lane_id=spawn_lane.id, path_length=min_path_length)
traffic_jam_3d_location = map_query.get_line_point_by_distance_from_start(
    line=star_vehicle_path, distance_from_start=distance_to_traffic
)

# Find the camera location, on the left edge of the road at a point closest to the star vehicle
camera_edge_line = map_query.get_edge_of_road_from_lane(lane_id=spawn_lane.id, side=Side.LEFT).as_polyline().to_numpy()
camera_location = camera_edge_line[np.argmin(np.linalg.norm(camera_edge_line - spawn_point, axis=1))]
camera_3d_position = np.array([camera_location[0], camera_location[1], camera_location[2] + camera_height])

# Create the ego vehicle with the StaticCameraBehavior
scenario.add_ego(
    CustomSimulationAgents.create_ego_vehicle(
        sensor_rig=sensor_rig,
        asset_name="",
        lock_to_ground=False,
    ).set_behaviour(
        LookAtPointBehavior(
            look_from=camera_3d_position,
            look_at=traffic_jam_3d_location,
        )
    )
)

# Add a Star vehicle that we can place the traffic jame relative to
scenario.add_agents(
    generator=VehicleGeneratorParameters(
        model="midsize_sedan_04",
        position_request=PositionRequest(
            absolute_position_request=AbsolutePositionRequest(
                position=Float3(
                    x=spawn_point[0],
                    y=spawn_point[1],
                ),
                resolve_z=True,
            )
        ),
        vehicle_spawn_data=VehicleSpawnData(
            vehicle_peripheral=VehiclePeripheral(
                disable_occupants=True,
            ),
            agent_spawn_data=AgentSpawnData(
                tags=[
                    "STAR",
                ]
            ),
        ),
    )
)

# Place a traffic jam ahead of the star vehicle
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        spawn_probability=1.0,
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["STAR"],
                max_spawn_radius=20.0,
            ),
            longitudinal_offset=CenterSpreadConfig(
                center=distance_to_traffic,
                spread=25,
            ),
        ),
        vehicle_spawn_data=VehicleSpawnData(
            vehicle_behavior=VehicleBehavior(
                start_speed=ContinousUniformDistribution(
                    min=0.0,
                    max=0.1,
                ),
                target_speed=ContinousUniformDistribution(
                    min=0.0,
                    max=0.1,
                ),
                lane_offset=ContinousUniformDistribution(
                    min=-0.2,
                    max=0.2,
                ),
                lane_drift_amplitude=ContinousUniformDistribution(
                    min=0.1,
                    max=0.3,
                ),
            )
        ),
    )
)

# Fill in the scene with other traffic
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        spawn_probability=1.0,
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["STAR"],
                max_spawn_radius=2 * distance_to_traffic,
            )
        ),
    )
)

preview_scenario(
    scenario=scenario,
    frames_per_scene=10,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
