import logging
from typing import Tuple

from pd.data_lab import LabelEngineInstance, RenderInstance, ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


# Our custom ScenarioCreator object
class IterationFlowExample(ScenarioCreator):
    # Method where we provide our scenario generation instructions
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Set up a simple sensor rig for demonstration purposes.  A simple 1920x1080 pinhole camera.
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

        # Create scenario object
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather variables for rain, fog and wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place our ego vehicle in the world
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

        # Place traffic in the world
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

        # Place parked cars in the world
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

    # Method to specify the Location and Lighting of the scenario
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # preview_scenario is the function which communicates with your Data Lab Instance.  Within this function, we specify
    # parameters such as the frames that we want returned, the frame rate of the scenario.

    # Note also that we specify the renderer and simulator separately.  This allows us to specify that we want to run
    # Data Lab in simulator-only mode. This can be done by passing in only the simulator parameter and not passing the
    # renderer and label_engine parameter.
    data_lab.preview_scenario(
        scenario_creator=IterationFlowExample(),  # The custom ScenarioCreator object which we created above
        random_seed=2023,  # A random seed which controls all random-based logic in our scenario generation instructions
        frames_per_scene=100,  # The number of frames we want to get back from our Data Lab Instance
        sim_capture_rate=10,  # The frame rate of the scenario. Defined as 100 / sim_capture_rate
        simulator=SimulationInstance(name="<instance name>"),  # Name of the instance we want to perform simulation on
        # Name of the instance we want to perform rendering and labeling on. Optional - if not provided, Data Lab
        # will run in simulation-only mode
        renderer=RenderInstance(name="<instance name>"),
        label_engine=LabelEngineInstance(name="<instance name>"),
    )
