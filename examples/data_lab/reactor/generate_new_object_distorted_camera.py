import logging
import tempfile
from typing import Callable, Optional

from pd.data_lab.config.distribution import Distribution, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Scenario
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION
from paralleldomain.data_lab.config.reactor import ReactorConfig, ReactorObject
from paralleldomain.data_lab.config.sensor_rig import DistortionParams
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.model.annotation import Depth, InstanceSegmentation2D, SemanticSegmentation2D
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

"""
Similar to the generate_new_object example, but with a strongly distorted fisheye. \
Reactor often creates better results on undistorted images, which can be enabled/disabled by setting undistort_input. \
Please note that this requires depth annotations to work!
"""


setup_loggers(logger_names=["__main__", "paralleldomain"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)


PROXY_OBJECT = "SM_NCAP_Male_Child_2yo_A_BobbyCar_01"  # used as an approximation for generated object's shape
OUTPUT_DATASET_PATH = tempfile.mkdtemp()
print(f"Output path is {OUTPUT_DATASET_PATH}")
setup_datalab(DEFAULT_DATA_LAB_VERSION)


class BlockEgoBehaviour(data_lab.CustomSimulationAgentBehaviour):
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

    def clone(self) -> "BlockEgoBehaviour":
        return BlockEgoBehaviour(
            vertical_offset=self.vertical_offset,
            dist_to_ego=self.dist_to_ego,
        )


cameras_to_use = [
    "Front",
]
sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig.create_camera_sensor(
            name="Front",
            width=1600,
            height=900,
            field_of_view_degrees=85.859,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 45.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
            annotation_types=[SemanticSegmentation2D, InstanceSegmentation2D, Depth],
            distortion_params=DistortionParams(
                **{
                    "cx": 730,
                    "cy": 410,
                    "fx": 860,
                    "fy": 861,
                    "k1": -0.4,
                    "k2": 0.0,
                    "k3": 0.0,
                    "k4": 0.0,
                    "k5": 0.0,
                    "k6": 0.0,
                    "is_fisheye": True,
                    "fisheye_model": 1,
                },
            ),
        ),
    ]
)
scenario = Scenario(sensor_rig=sensor_rig)
scenario.random_seed = (
    128  # random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic
)

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
    data_lab.CustomSimulationAgents.create_object(asset_name=PROXY_OBJECT, lock_to_ground=True).set_behaviour(
        BlockEgoBehaviour(
            dist_to_ego=Distribution.create(value=10.0).sample(random_seed=scenario.random_seed),
            vertical_offset=Distribution.create(value=(-0.2, 0.2)).sample(random_seed=scenario.random_seed),
        )
    )
)


reactor_object = ReactorObject(prompts=["bobby car"], asset_name=PROXY_OBJECT, new_class_id=200)
reactor_config = ReactorConfig(
    reactor_object=reactor_object,
    cameras_to_use=cameras_to_use,
    undistort_input=True,
)


data_lab.create_mini_batch(
    scenario=scenario,
    frames_per_scene=4,
    number_of_scenes=2,
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
