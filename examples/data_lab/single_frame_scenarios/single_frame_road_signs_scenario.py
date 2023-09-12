import logging

from pd.data_lab.context import setup_datalab, load_map
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION
from paralleldomain.data_lab.config.map import LaneSegment, MapQuery
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.road_signs import SignGenerator
from paralleldomain.data_lab.generators.single_frame import (
    SingleFrameEgoGenerator,
    SingleFrameNonEgoVehicleGenerator,
    SingleFrameVehicleBehaviorType,
)
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab(DEFAULT_DATA_LAB_VERSION)

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
                yaw=00.0,
                x=0.0,
                y=0.0,
                z=2.0,
            ),
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = 121697  # random.randint(1, 1000000) # keep integer fixed for deterministic scenario generation

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
location = data_lab.Location(name="SF_6thAndMission_medium")
scenario.set_location(location=location)

umd_map = load_map(location=location)
map_query = MapQuery(umd_map)

# Place ourselves in the world
scenario.add_ego(
    generator=SingleFrameEgoGenerator(
        lane_type=LaneSegment.LaneType.DRIVABLE,
        ego_asset_name="suv_medium_02",
        random_seed=scenario.random_seed,
        sensor_rig=sensor_rig,
    )
)

scenario.add_agents(
    generator=SingleFrameNonEgoVehicleGenerator(
        number_of_vehicles=10,
        random_seed=scenario.random_seed,
        spawn_radius=50.0,
        vehicle_behavior_type=SingleFrameVehicleBehaviorType.TRAFFIC,
    )
)

scenario.add_agents(
    generator=SignGenerator(
        num_sign_poles=30,
        max_signs_per_pole=3,
        country="Portugal",
        radius=40.0,
        forward_offset_to_place_signs=40.0,
        min_distance_between_signs=1.5,
        single_frame_mode=True,
        random_seed=scenario.random_seed,
    )
)

data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=10,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
