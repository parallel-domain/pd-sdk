import itertools
import logging

from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

setup_datalab("v2.1.0-beta")


sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig.create_camera_sensor(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
        )
    ]
)

WEATHER_INTENSITIES = [0.0, 0.5, 1.0]
TIMES_OF_DAY = [data_lab.TimeOfDays.Day, data_lab.TimeOfDays.Dawn, data_lab.TimeOfDays.Dusk, data_lab.TimeOfDays.Night]
LOCATIONS = [
    "SF_6thAndMission_medium",
    "A2_BurnsPark",
    "A2_Kerrytown",
    "SC_Highlands",
    "SC_MathildaAndSunnyvaleSaratoga_large",
    "SC_W8thAndOrchard",
    "SF_GrantAndCalifornia",
    "SF_JacksonAndKearny",
    "SF_VanNessAveAndTurkSt",
    "SJ_237AndGreatAmerica",
    "SJ_237AndZanker",
    "SJ_680MissionPass",
    "SJ_EssexAndBradford",
    "SJ_KettmanAndOrinda_aus",
]

for tod in TIMES_OF_DAY:  # 4 time of day groups
    for i, j, k, l in itertools.product(WEATHER_INTENSITIES, repeat=4):  # 81 weather combinations
        # Create scenario
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)
        scenario.random_seed = 1337  # set to a fixed integer to keep scenario generation deterministic

        # Set weather variables and time of day - 45 tests
        # default day
        scenario.environment.time_of_day.set_category_weight(tod, 1.0)
        scenario.environment.clouds.set_constant_value(i)
        scenario.environment.rain.set_constant_value(j)
        scenario.environment.wetness.set_constant_value(k)
        scenario.environment.fog.set_constant_value(l)

        # Select an environment
        scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.1.0-beta"))

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

        # scenario.save_scenario(path=AnyPath("my_scenario.json"))

        output_directory = AnyPath("weather_test_output")
        output_directory.mkdir(exist_ok=True, parents=True)
        for frame, scene in data_lab.create_frame_stream(
            scenario=scenario,
            frames_per_scene=1,
            number_of_scenes=1,
            sim_settle_frames=1,
            sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:30XX"),
            render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:30XX"),
        ):
            for camera_frame in frame.camera_frames:
                write_png(
                    obj=camera_frame.image.rgb,
                    path=AnyPath(f"out/{camera_frame.sensor_name}_{camera_frame.frame_id:0>18}.png"),
                )
                write_png(
                    obj=camera_frame.image.rgb,
                    path=output_directory
                    / f"{camera_frame.sensor_name}_{camera_frame.frame_id}_{tod.value}_{i}_{j}_{k}_{l}.png",
                )
                logger.info(f"Complete scene {tod.value} - clouds: {i} rain: {j}, wetness: {k}, fog: {l}")
