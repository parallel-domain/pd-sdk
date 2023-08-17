import logging

from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters, RenderEgoGenerator
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.4.0-beta")

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
                yaw=70.0,
                x=4.0,
                y=0.0,
                z=0.5,
            ),
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = 1000  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Dusk, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
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
                )
            )
        ),
    ),
)

scenario.add_agents(
    generator=RenderEgoGenerator(
        ego_asset_name="suv_medium_02",
    ),
)


data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=20,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
