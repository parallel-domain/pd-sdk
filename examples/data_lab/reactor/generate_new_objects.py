import logging
import random
import tempfile
from typing import Callable, Optional

import numpy as np
from pd.data_lab.config.distribution import Distribution, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Scenario
from pd.data_lab.sim_instance import SimulationInstance
from pd.sim import Raycast

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.reactor import ReactorConfig, ReactorObject
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.model.annotation import InstanceSegmentation2D, SemanticSegmentation2D
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)


PROXY_OBJECT = "SM_primitive_box_1m"  # used as an approximation for generated object's shape
PROXY_SCALE_FACTORS = [0.6, 0.4, 0.8]  # L*W*H
OUTPUT_DATASET_PATH = tempfile.mkdtemp()
print(f"Output path is {OUTPUT_DATASET_PATH}")
setup_datalab("v2.4.1-beta")


class BlockEgoBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        vertical_offset: float = 0.0,
        dist_to_ego: float = 5.0,
        scale_factors: np.array = np.array([1.0, 1.0, 1.0]),
    ):
        super().__init__()
        self.dist_to_ego = dist_to_ego
        self.vertical_offset = vertical_offset
        self.scale_factors = scale_factors

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

        # Scale the object
        pose_with_scale = pose.transformation_matrix
        pose_with_scale[0, 0] = self.scale_factors[0]
        pose_with_scale[1, 1] = self.scale_factors[1]
        pose_with_scale[2, 2] = self.scale_factors[2]

        # Find position on ground
        start_pos = pose.translation + data_lab.coordinate_system.up
        result = raycast(
            [Raycast(origin=tuple(start_pos), direction=tuple(data_lab.coordinate_system.down), max_distance=10)]
        )
        if len(result[0]) > 0:
            pose_with_scale[:3, 3] = result[0][0].position

        agent.set_pose(pose=pose_with_scale)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "BlockEgoBehaviour":
        return BlockEgoBehaviour(
            vertical_offset=self.vertical_offset,
            dist_to_ego=self.dist_to_ego,
            scale_factors=self.scale_factors,
        )


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
            annotation_types=[SemanticSegmentation2D, InstanceSegmentation2D],
        )
    ]
)

# Create scenario
scenario = Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
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
    data_lab.CustomSimulationAgents.create_object(asset_name=PROXY_OBJECT, lock_to_ground=False).set_behaviour(
        BlockEgoBehaviour(
            dist_to_ego=Distribution.create(value=10.0).sample(random_seed=scenario.random_seed),
            vertical_offset=Distribution.create(value=(-0.2, 0.2)).sample(random_seed=scenario.random_seed),
            scale_factors=np.array(PROXY_SCALE_FACTORS),
        )
    )
)


reactor_object = ReactorObject(prompts=["bobby car", "red bobby car"], asset_name=PROXY_OBJECT, new_class_id=200)
reactor_config = ReactorConfig(reactor_object=reactor_object)
reactor_config.use_color_matching = True
reactor_config.inference_resolution = 512


data_lab.create_mini_batch(
    scenario=scenario,
    frames_per_scene=3,
    number_of_scenes=2,
    show_image_for_n_seconds=5,
    sim_capture_rate=10,
    start_skip_frames=5,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
    reactor_config=reactor_config,
    use_cached_reactor_states=True,
    format_kwargs=dict(
        dataset_output_path=OUTPUT_DATASET_PATH,
        encode_to_binary=False,
    ),
    pipeline_kwargs=dict(copy_all_available_sensors_and_annotations=True, run_env="thread"),
)
