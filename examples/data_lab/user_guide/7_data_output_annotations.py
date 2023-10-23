import logging
import random
from typing import List, Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import create_frame_stream
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.visualization.model_visualization import show_frame

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


def images_to_disk(
    scenario_creator: ScenarioCreator,
    storage_path: AnyPath,
    random_seed: int,
    number_of_scenes: int = 1,
    frames_per_scene: int = 10,
    annotations_to_show: List[AnnotationType] = None,
    **kwargs,
):
    # This function is a custom function that saves the rendered images to disk. It follows the same structure as the
    # preview_scenario function, but instead of visualizing the images, it saves them to disk.
    # The function has been defined here for demonstration purposes, but it can be copied and used in your own code.
    # It uses the `create_frame_stream` function to create a stream of frames, and then iterates over the frames and
    # saves the rendered images to disk.
    # This function can be easily extended to save annotations to disk as well.
    storage_path.mkdir(exist_ok=True, parents=True)  # create output directory if not exists
    scene_indices = list(range(number_of_scenes))
    for frame, scene in create_frame_stream(
        scenario_creator=scenario_creator,
        frames_per_scene=frames_per_scene,
        scene_indices=scene_indices,
        number_of_scenes=number_of_scenes,
        random_seed=random_seed,
        **kwargs,
    ):
        show_frame(frame=frame, annotations_to_show=annotations_to_show)
        for camera_frame in frame.camera_frames:
            write_png(
                obj=camera_frame.image.rgb,
                path=storage_path / f"{camera_frame.sensor_name}_{camera_frame.frame_id:0>18}.png",
            )


# Create our custom ScenarioCreator class
class OutputAndAnnotationsExample(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Create a simple pinhole sensor rig for demo purposes
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

        # Create a scenario object
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set the weather of the scenario
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.5, max_value=0.9)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.2, max_value=0.6)

        # Place the ego vehicle in the world using Pre-Built Generators
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

        # Place traffic in the scenario using Pre-Built Generators
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

        # Place parked vehicles in the scenario using Pre-Built Generators
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
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # Up until now, we have used the preview_scenario function to communicate with the Data Lab Instance.
    # This allows the generated scenario to be visualized in the Data Lab Visualizer, including both the rendered image
    # and the wireframe representation of the state.
    data_lab.preview_scenario(
        scenario_creator=OutputAndAnnotationsExample(),
        random_seed=2023,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance name>",
    )

    # Another available option is our custom image_to_disk function which saves the rendered images to disk without
    # visualizing them. The function has been defined at the top of this file.
    # images_to_disk(
    #     scenario_creator=OutputAndAnnotationsExample(),
    #     random_seed=2023,
    #     frames_per_scene=100,
    #     sim_capture_rate=10,
    #     storage_path=AnyPath("output"),
    #     instance_name="<instance name>",
    # )
