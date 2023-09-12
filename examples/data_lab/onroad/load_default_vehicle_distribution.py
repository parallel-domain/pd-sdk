import logging
import random
from typing import Dict

from pd.assets import DataVehicle, DataVehicleTypeSpawnChance, ObjAssets, UtilVehicleTypes
from pd.data_lab.config.distribution import EnumDistribution, VehicleCategoryWeight
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab(DEFAULT_DATA_LAB_VERSION)


def query_vehicles_with_spawn_probability():
    query_result = (  # query the Asset DB to find vehicle categories, their spawn chance and their associated vehicles
        DataVehicle.select(
            DataVehicle.spawn_chance,
            ObjAssets.name,
            UtilVehicleTypes.name,
            DataVehicleTypeSpawnChance.spawn_chance,
        )
        .join(ObjAssets, on=(DataVehicle.asset_id == ObjAssets.id))
        .switch(DataVehicle)
        .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type == UtilVehicleTypes.id))
        .join(DataVehicleTypeSpawnChance, on=(DataVehicleTypeSpawnChance.vehicle_type_id == UtilVehicleTypes.id))
    )

    return query_result


def default_vehicle_distribution() -> Dict[str, VehicleCategoryWeight]:
    vehicle_weights = query_vehicles_with_spawn_probability()

    vehicle_distribution = {
        vehicle_type.vehicle_type.name: VehicleCategoryWeight(
            weight=vehicle_type.vehicle_type.datavehicletypespawnchance.spawn_chance, model_weights={}
        )
        for vehicle_type in vehicle_weights.group_by(UtilVehicleTypes.name)
    }

    for vehicle in vehicle_weights:
        vehicle_distribution[vehicle.vehicle_type.name].model_weights[vehicle.asset.name] = vehicle.spawn_chance

    return vehicle_distribution


vehicle_weights = default_vehicle_distribution()

sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig.create_camera_sensor(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium"))

# Place ourselves in the world
scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            lane_spawn_policy=LaneSpawnPolicy(
                lane_type=EnumDistribution(
                    probabilities={"Drivable": 1.0},
                ),
            )
        ),
    ),
)

# # Place other agents
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
            ),
        ),
        spawn_probability=0.8,
        vehicle_distribution=vehicle_weights,
    )
)

data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
