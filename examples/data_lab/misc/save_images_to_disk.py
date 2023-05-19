import logging
import random
from typing import List

from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Scenario
from pd.data_lab.sim_instance import SimulationInstance
from pd.internal.proto.keystone.generated.wrapper.pd_sensor_pb2 import CameraIntrinsic, SensorExtrinsic

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.visualization.sensor_frame_viewer import show_sensor_frame

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.1.0-beta")


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
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

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


AnyPath("out").mkdir(exist_ok=True, parents=True)  # create output directory if not exists


def preview_scenario(
    scenario: Scenario,
    number_of_scenes: int = 1,
    frames_per_scene: int = 10,
    annotations_to_show: List[AnnotationType] = None,
    **kwargs,
):
    for frame, scene in data_lab.create_frame_stream(
        scenario=scenario, frames_per_scene=frames_per_scene, number_of_scenes=number_of_scenes, **kwargs
    ):
        for camera_frame in frame.camera_frames:
            write_png(
                camera_frame.image.rgb, AnyPath(f"out/{camera_frame.sensor_name}_{camera_frame.frame_id:0>18}.png")
            )
            show_sensor_frame(sensor_frame=camera_frame, annotations_to_show=annotations_to_show, frames_per_second=100)


preview_scenario(
    scenario=scenario,
    frames_per_scene=500,
    sim_capture_rate=1,
    sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:30XX"),
    render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:30XX"),
)
