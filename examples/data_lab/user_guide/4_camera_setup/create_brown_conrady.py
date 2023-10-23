import logging
from typing import Tuple

from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.scenario import Lighting, ScenarioCreator, ScenarioSource

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
    This script goes through a demonstration of how to set up an Brown Conrady Camera in Data Lab.  Data Lab uses the
        OpenCV Pinhole model (fisheye_model=0) to replicate Brown Conrady Distortion.

    In setting up a Brown Conrady cameras, we must pass in the following parameters
        name - Name of the camera
        width - The pixel width of the image
        height - The pixel height of the image
        pose - Transformation object containing the position and rotation of the camera
        distortion_params - A DistortionParams object that contains the OpenCV Distortion Parameters as follows:
            k1,k2,p1,p2[,k3[,k4,k5,k6]] - Distortion parameters according to the OpenCV Pinhole Documentation for
                projection and distortion functions as documented at
                https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html.  Parameters in brackets are optional
            fx - Horizontal direction focal length
            fy - Vertical direction focal length
            cx - Horizontal direction distortion center
            cy - Vertical direction distortion center
            fisheye_model = The model of camera to be replicated (should be fisheye_model=0 for OpenCV Brown Conrady
                and OpenCV Pinhole)


    In this example script, we will be demonstrating one example of a camera with Brown Conrady distortion:
        Brown_Conrady_Front - A 1920 x 1080 Brown Conrady distortion camera with the distortion parameters listed below
"""

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


# Set up our custom ScenarioCreator object
class BrownConradyCam(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # Set up our Brown Conrady camera
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Brown_Conrady_Front",
            width=1920,
            height=1080,
            pose=Transformation.from_euler_angles(
                angles=[-10.0, 0.0, 0.0],
                order="xyz",
                translation=[0.0, 0.0, 1.5],
                degrees=True,
            ),
            distortion_params=DistortionParams(
                fx=1662.8,
                fy=1662.8,
                cx=960,
                cy=540,
                k1=-0.35,
                k2=0.07,
                k3=-0.002,
                k4=0.0006,
                k5=0.0,
                k6=0.0,
                p1=0.0001,
                p2=-0.0002,
                fisheye_model=0,
            ),
        )

        # Create our Scenario object
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather in scenario
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place ego agent
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

        # Place traffic in the scenario
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

        # Place parked vehicle in the scenario
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
        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # Render our scenario
    data_lab.preview_scenario(
        scenario_creator=BrownConradyCam(),
        frames_per_scene=10,
        sim_capture_rate=10,
        random_seed=1000,
        instance_name="<instance name>",
    )
