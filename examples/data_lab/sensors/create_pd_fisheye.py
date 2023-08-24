import logging

from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import DistortionParams
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

"""
    This script goes through a demonstration of how to set up a PD Fisheye Camera in Data Lab.  The PD Fisheye Model
        (fisheye_model=3) uses a custom PD Look Up Table for representing radial lens distortion

    In setting up a PD Fisheye camera, we must pass in the following parameters
        name - Name of the camera
        width - The pixel width of the image
        height - The pixel height of the image
        pose - Transformation object containing the position and rotation of the camera
        distortion_params - A DistortionParams object that contains the Distortion Parameters as follows:
            fx - Horizontal direction focal length
            fy - Vertical direction focal length
            cx - Horizontal direction distortion center
            cy - Vertical direction distortion center
            fisheye_model = The model of camera to be replicated (should be 3 for PD Fisheye)
        distortion_lookup_table - A filepath to the PD Look Up table that represents the radial distortion of the
            camera.  This filepath will be provided by your Technical Customer Success Manager

    In this example script, we will be demonstrating one example of a PD Fisheye cameras:
        PD_Fisheye_Front - A 1920 x 1080 PD Fisheye distortion camera facing frontward with a custom PD Distortion
            Look Up Table
"""

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.4.1-beta")

sensor_rig = data_lab.SensorRig().add_camera(
    name="PD_Fisheye_Front",
    width=1920,
    height=1080,
    pose=Transformation.from_euler_angles(
        angles=[0.0, 0.0, 0.0],
        order="xyz",
        translation=[0.0, 0.0, 1.5],
        degrees=True,
    ),
    distortion_params=DistortionParams(
        fx=500.0,
        fy=500.0,
        cx=960,
        cy=540,
        fisheye_model=3,
    ),
    distortion_lookup_table="customer\\Internal\\sensor_rigs\\product_PDRadialFisheye_Front.csv",
)

scenario = data_lab.Scenario(sensor_rig=sensor_rig)
# scenario.random_seed = random.randint(1, 1_000_000)
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
        vehicle_spawn_data=VehicleSpawnData(
            vehicle_peripheral=VehiclePeripheral(
                disable_occupants=True,
            )
        ),
    ),
)

# Place other agents
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        spawn_probability=0.4,
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
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
                max_spawn_radius=100.0,
            )
        ),
    )
)


data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=10,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
