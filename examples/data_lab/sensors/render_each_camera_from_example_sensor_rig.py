import logging
from typing import List

# imports to configure generators
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.context import setup_datalab

# import to enable preview scenario
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

# imports to configure sensors
from pd.internal.proto.keystone.generated.wrapper.pd_sensor_pb2 import (
    CameraIntrinsic,
    DistortionParams,
    SensorExtrinsic,
)

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
    SpecialAgentTag,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.annotation import AnnotationType

# imports to configure output
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.visualization.sensor_frame_viewer import show_sensor_frame

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.1.0-beta")

all_sensors = [
    data_lab.SensorConfig(
        display_name="PinholeFOV_Front",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1080,
            fov=70,
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            x=-0.0,
            y=-0.7,
            z=1.5,
        ),
    ),
    data_lab.SensorConfig(
        display_name="PinholeFOV_Left",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1080,
            fov=90,
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=0.0,
            yaw=90.0,
            x=-0.65,
            y=-0.5,
            z=1.5,
        ),
    ),
    data_lab.SensorConfig(
        display_name="PinholeFOV_Right",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1080,
            fov=90,
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=0.0,
            yaw=-90.0,
            x=0.65,
            y=-0.5,
            z=1.5,
        ),
    ),
    data_lab.SensorConfig(
        display_name="PinholeFOV_Rear",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1080,
            fov=90,
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=0.0,
            yaw=180.0,
            x=0.0,
            y=-2.30,
            z=1.5,
        ),
    ),
    data_lab.SensorConfig(
        display_name="OpenCVBrownConrady_Front",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1080,
            distortion_params=DistortionParams(
                fx=1662.7687752661225,
                fy=1662.7687752661225,
                cx=960.0,
                cy=540.0,
                k1=-0.35,
                k2=0.07,
                k3=-0.002,
                k4=0.0006,
                k5=0.0,
                k6=0.0,
                p1=0.0001,
                p2=-0.0002,
                skew=0.0,
                fisheye_model=0,
            ),
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=-10.0,
            yaw=0.0,
            x=0.0,
            y=0.7,
            z=1.5,
        ),
    ),
    data_lab.SensorConfig(
        display_name="Ortho_BEV",
        camera_intrinsic=CameraIntrinsic(
            width=1920,
            height=1920,
            distortion_params=DistortionParams(
                fx=19.2,
                fy=19.2,
                cx=960.0,
                cy=960.0,
                p1=-200,
                p2=300,
                skew=0,
                fisheye_model=6,
            ),
        ),
        sensor_extrinsic=SensorExtrinsic(
            roll=0.0,
            pitch=-90.0,
            yaw=0.0,
            x=0.0,
            y=0.0,
            z=100,
        ),
    ),
]


for sensor in all_sensors:
    scenario = data_lab.Scenario(sensor_rig=data_lab.SensorRig(sensor_configs=[sensor]))
    scenario.random_seed = 1337

    # Set weather variables and time of day
    scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
    scenario.environment.clouds.set_constant_value(0.5)
    scenario.environment.rain.set_constant_value(0.0)
    scenario.environment.fog.set_constant_value(0.2)
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
            vehicle_spawn_data=VehicleSpawnData(
                vehicle_peripheral=VehiclePeripheral(
                    disable_occupants=True
                )  # disable occupants so our cameras can see through them
            ),
        ),
    )

    # Place other agents
    scenario.add_agents(
        generator=TrafficGeneratorParameters(
            spawn_probability=0.8,
            position_request=PositionRequest(
                location_relative_position_request=LocationRelativePositionRequest(
                    agent_tags=[SpecialAgentTag.EGO],
                    max_spawn_radius=100.0,
                )
            ),
        )
    )

    scenario.add_agents(
        generator=ParkedVehicleGeneratorParameters(
            spawn_probability=CenterSpreadConfig(center=0.6),
            position_request=PositionRequest(
                location_relative_position_request=LocationRelativePositionRequest(
                    agent_tags=[SpecialAgentTag.EGO],
                    max_spawn_radius=100.0,
                )
            ),
        )
    )

    def preview_scenario(
        scenario,
        number_of_scenes: 1,
        frames_per_scene: 1,
        annotations_to_show: List[AnnotationType] = None,
        **kwargs,
    ):
        AnyPath("sensor_test_output").mkdir(exist_ok=True)
        for frame, scene in data_lab.create_frame_stream(
            scenario=scenario, frames_per_scene=frames_per_scene, number_of_scenes=number_of_scenes, **kwargs
        ):
            for camera_frame in frame.camera_frames:
                write_png(
                    obj=camera_frame.image.rgb,
                    path=AnyPath(f"sensor_test_output/{camera_frame.sensor_name}.png"),
                )
                show_sensor_frame(
                    sensor_frame=camera_frame, annotations_to_show=annotations_to_show, frames_per_second=100
                )

    preview_scenario(
        scenario=scenario,
        frames_per_scene=1,
        number_of_scenes=1,
        sim_settle_frames=100,
        sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:30XX"),
        render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:30XX"),
    )
