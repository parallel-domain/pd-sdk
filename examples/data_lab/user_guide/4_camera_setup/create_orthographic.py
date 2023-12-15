import logging
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import DistortionParams
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

"""
    This script goes through a demonstration of how to set up orthographic cameras in Data Lab.

    As seen below, orthographic cameras in Data Lab are created the same was as other cameras, but are
    represented by fisheye_model=6

    In setting up orthographic cameras, we must pass in the following camera intrinsic parameters and
    Distortion parameters with the definitions listed below:
        width - The pixel width of the image.  Combines with fx to determine FOV in x direction
        height - The pixel height of the image.  Combines with fy to determine FOV in y direction
        fx - pixel/m resolution of the image in the x direction
        fy - pixel/m resolution of the image in the y direction
        p1 - Near clip plane of the orthographic camera
        p2 - Far clip plane of the orthographic camera

    In this example script, we will be demonstrating three examples of orthographic cameras:
        Ortho_BEV - Square Orthographic BEV Camera 100m x 100m in size and 19.2 pixels/m resolution
        Ortho_BEV_low_res - Square Orthographic BEV Camera 100m x 100m in size and 1/5th the resolution of Ortho_BEV
        Ortho_BEV_low_fov - Square Orthographic BEV Camera 20m x 20m in size and 19.2 pixels/m resolution
"""

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


# Set up our custom ScenarioCreator object
class OrthographicCamera(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # BEV Ortho Camera Parameters
        bev_camera_fov = 100  # Camera FOV in meters (square so height=width)
        bev_base_resolution = 19.2  # Resolution of Ortho_BEV camera in pixels/m
        bev_near_clip_plane = -200
        bev_far_clip_plane = 300

        # Set up our Orthographic BEV camera
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Ortho_BEV",
            width=int(bev_camera_fov * bev_base_resolution),
            height=int(bev_camera_fov * bev_base_resolution),
            pose=Transformation.from_euler_angles(
                angles=[-90.0, 0.0, 0.0],
                order="xyz",
                translation=[0.0, 0.0, 100],
                degrees=True,
            ),
            distortion_params=DistortionParams(
                fx=bev_base_resolution,
                fy=bev_base_resolution,
                p1=bev_near_clip_plane,
                p2=bev_far_clip_plane,
                fisheye_model=6,
            ),
        )

        # Add a lower resolution version of the orthographic BEV camera to the sensor rig
        sensor_rig.add_camera(
            name="Ortho_BEV_low_res",
            width=int(bev_camera_fov * bev_base_resolution / 5),
            height=int(bev_camera_fov * bev_base_resolution / 5),
            pose=Transformation.from_euler_angles(
                angles=[-90.0, 0.0, 0.0],
                order="xyz",
                translation=[0.0, 0.0, 100],
                degrees=True,
            ),
            distortion_params=DistortionParams(
                fx=bev_base_resolution / 5,
                fy=bev_base_resolution / 5,
                p1=bev_near_clip_plane,
                p2=bev_far_clip_plane,
                fisheye_model=6,
            ),
        )

        # Add another orthographic BEV camera with a lower field of view
        sensor_rig.add_camera(
            name="Ortho_BEV_low_fov",
            width=int(bev_camera_fov / 5 * bev_base_resolution),
            height=int(bev_camera_fov / 5 * bev_base_resolution),
            pose=Transformation.from_euler_angles(
                angles=[-90.0, 0.0, 0.0],
                order="xyz",
                translation=[0.0, 0.0, 100],
                degrees=True,
            ),
            distortion_params=DistortionParams(
                fx=bev_base_resolution,
                fy=bev_base_resolution,
                p1=bev_near_clip_plane,
                p2=bev_far_clip_plane,
                fisheye_model=6,
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
            ),
        )

        # Place traffic in the scenario
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

        # Place parked vehicle in the scenario
        scenario.add_agents(
            generator=ParkedVehicleGeneratorParameters(
                spawn_probability=CenterSpreadConfig(center=0.6),
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
        scenario_creator=OrthographicCamera(),
        frames_per_scene=10,
        sim_capture_rate=10,
        random_seed=1000,
        instance_name="<instance name>",
    )
