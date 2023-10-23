import dataclasses
from typing import Callable, List, Optional, Tuple

from pd.data_lab.config.distribution import Distribution, EnumDistribution
from pd.data_lab.scenario import Lighting, ScenarioCreator, ScenarioSource

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


class BlockEgoBehavior(data_lab.CustomSimulationAgentBehavior):
    def __init__(self, vertical_offset: float = 0.0, dist_to_ego: float = 5.0):
        super().__init__()
        self.dist_to_ego = dist_to_ego
        self.vertical_offset = vertical_offset

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        pos_in_ego_coords = data_lab.coordinate_system.forward * self.dist_to_ego
        vert_offset = data_lab.coordinate_system.left * self.vertical_offset
        pos_in_ego_coords += vert_offset
        pose_in_front_of_ego = sim_state.ego_pose @ pos_in_ego_coords
        pose = Transformation(translation=pose_in_front_of_ego)
        agent.set_pose(pose=pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "BlockEgoBehavior":
        return BlockEgoBehavior(vertical_offset=self.vertical_offset, dist_to_ego=self.dist_to_ego)


@dataclasses.dataclass
class CustomObstacleGenerator(data_lab.CustomAtomicGenerator):
    number_of_agents: Distribution = Distribution.create(value=1)
    distance_to_ego: Distribution = Distribution.create(value=10.0)
    vertical_offset: Distribution = Distribution.create(value=(-0.2, 0.2))

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []
        for _ in range(int(self.number_of_agents.sample(random_seed=random_seed))):
            agent = data_lab.CustomSimulationAgents.create_object(asset_name="portapotty_01").set_behavior(
                BlockEgoBehavior(
                    dist_to_ego=self.distance_to_ego.sample(random_seed=random_seed),
                    vertical_offset=self.vertical_offset.sample(random_seed=random_seed),
                )
            )
            agents.append(agent)
        return agents

    def clone(self):
        return CustomObstacleGenerator(
            number_of_agents=self.number_of_agents.clone(),
            distance_to_ego=self.distance_to_ego.clone(),
            vertical_offset=self.vertical_offset.clone(),
        )


class BlockingObject(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Front",
            width=768,
            height=768,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
            annotation_types=[AnnotationTypes.SemanticSegmentation2D],
        )

        # Create scenario
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather variables and time of day
        scenario.environment.rain.set_constant_value(0.1)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

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

        # Add other agents
        # Place other agents
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["EGO"],
                        max_spawn_radius=200.0,
                    )
                ),
            )
        )

        scenario.add_agents(CustomObstacleGenerator())

        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=BlockingObject(),
        random_seed=133,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance name>",
    )
