# Data Generation - Getting Started

## Setup

First of you'll need to install the generation dependencies

```bash
pip install "paralleldomain[data_lab,visualization] @ git+ssh://git@github.com/parallel-domain/pd-sdk-internal@main#egg=paralleldomain[data_lab,visualization]"
```


Secondly you need to set you credentials. For this run

``pd-credentials-setup``

which will prompt you to enter you step username, password and path to your .pem file

## Define a Scenario with PD Simulation in the Loop

```python
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.transformation import Transformation
import paralleldomain.data_lab as data_lab
import dataclasses

# First define a sensor rig. Assume a FLU (x = Front y = Left z = Up) Coordinate system.
# Here we define a sensor rig with 2 cameras. The First camera captures Semantic Segmentation Labels.
# The second camera does not and only runs at 4Hz while the first one runs at 2Hz.


sensor_rig = (
    data_lab.SensorRig()
    .add_camera(
        name="Front",
        width=768,
        height=768,
        frame_rate=4.0,
        field_of_view_degrees=70,
        pose=Transformation.from_euler_angles(
            angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
        ),
        annotation_types=[AnnotationTypes.SemanticSegmentation2D],
    )
    .add_camera(
        name="Rear",
        width=768,
        height=768,
        frame_rate=2.0,
        field_of_view_degrees=70,
        pose=Transformation.from_euler_angles(
            angles=[0.0, 0.0, 180.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
        ),
        annotation_types=[],
    )
)

# Create a Scenario and pass the sensor rig. A Scenario is a collection of distributions that result in discrete Scenes.

scenario = data_lab.Scenario(sensor_rig=sensor_rig)

# Here we define the distribution of the environments variables we want to see in our Scenes

# Here we only want to see Day or Night Scenes. Therefore we set the weight of each Category (Day and Night)
# to the same value
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Night, 1.0)

# We set the cloud density and the rain intensity to fixed values (in [0, 1]) for all Scenes
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.1)

# The street wetness will be sampled from a uniform distribution within [0.1, 0.3]
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# We want an Urban map from the PD catalogue called "SF_6thAndMission_medium"
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.0.1"))

# If you want PD to control some of the agents in the Scene you can add predefined Generators to controll those
# You can stack arbitray Genertors on top of each other to combine different behaviours.

# Lets say we want PD to control Vehicles and Pedestrians. Here we add one generator for vehicles within 100m of the ego
# and one for pedestrians also within 100m. Here we will get between 30 and 45 PEdestrians in each Scene.
scenario.add_agents(data_lab.VehicleTrafficGenerator.create_near_ego(density=0.3, radius_to_ego=100.0))
scenario.add_agents(data_lab.RandomPedestrianGenerator.create_near_ego(count_min=30, count_max=45, radius_to_ego=100.0))

# Lets also have our ego driven by PD
scenario.add_ego(generator=data_lab.EgoVehicleGenerator())


# And now lets write our own Generator that places Porta Potties in front of the Ego.


class BlockEgoBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(self, vertical_offset: float = 0.0, dist_to_ego: float = 5.0):
        super().__init__()
        self.dist_to_ego = dist_to_ego
        self.vertical_offset = vertical_offset

    def set_inital_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        pos_in_ego_coords = data_lab.SimCoordinateSystem.get_local_forward_direction() * self.dist_to_ego
        vert_offset = data_lab.SimCoordinateSystem.get_local_left_direction() * self.vertical_offset
        pos_in_ego_coords += vert_offset
        pose_in_front_of_ego = sim_state.ego_pose @ pos_in_ego_coords
        pose = Transformation(translation=pose_in_front_of_ego)
        agent.set_pose(pose=pose)

    def update_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        # Since we dont want the Porta Potty to move, we dont update them
        pass


@dataclasses.dataclass
class MyObstacleGenerator(data_lab.CustomAtomicGenerator):
    number_of_agents: data_lab.RandomVariable[int] = data_lab.RandomVariable(default=4)
    distance_to_ego: data_lab.RandomVariable[float] = data_lab.RandomVariable(default=10.0)
    vertical_offset: data_lab.RandomVariable[float] = data_lab.RandomVariable(default=(-0.2, 0.2))

    def create_agents_for_new_scene(self, state: data_lab.SimState):
        agents = []
        for _ in range(int(self.number_of_agents.sample())):
            # "portapotty_01"
            agent = data_lab.SimAgents.create_object(asset_name="portapotty_01").set_behaviour(
                BlockEgoBehaviour(
                    dist_to_ego=self.distance_to_ego.sample(), vertical_offset=self.vertical_offset.sample()
                )
            )
            agents.append(agent)
        return agents


# Now we add our custom generator to the scenario
scenario.add_agents(MyObstacleGenerator())

# Now we can choose to preview our scenario, encode it to disk or s3, or to create a live data stream

# preview
data_lab.preview_scenario(
    scenario=scenario,
    dataset_name="test",
    number_of_scenes=1,
    frames_per_scene=5,  # controls the render quality. Higher is better, but slower
    pd_sim_address="tcp://35.87.67.84:9002",  # You need a running sim state instance if you want pd generators
)

# encode

data_lab.create_mini_batch(
    scenario=scenario,
    frames_per_scene=10,
    number_of_scenes=2,
    pd_sim_address="tcp://35.87.67.84:9002",  # You need a running sim state instance if you want pd generators
    dataset_name="test",
    format_kwargs=dict(
        dataset_output_path="/tmp/debug-mini4/",
        encode_to_binary=True,
    ),
    pipeline_kwargs=dict(copy_all_available_sensors_and_annotations=True),
)

# stream and encode

for pipeline_item in data_lab.create_mini_batch_stream(
    scenario=scenario,
    frames_per_scene=5,
    number_of_scenes=1,
    pd_sim_address="tcp://35.87.67.84:9002",  # You need a running sim state instance if you want pd generators
):
    camera_frame = pipeline_item.camera_frame
    if camera_frame is not None:
        cv2.imshow("image", camera_frame.image.rgb[..., ::-1])
        cv2.waitKey(1000)

# just stream

for frame in data_lab.create_frame_stream(
    scenario=scenario,
    frames_per_scene=5,
    number_of_scenes=1,
    pd_sim_address="tcp://35.87.67.84:9002",  # You need a running sim state instance if you want pd generators
):
    for camera_frame in frame.camera_frames:
        cv2.imshow("image", camera_frame.image.rgb[..., ::-1])
        cv2.waitKey(1000)
```


## Define scenario with only custom Generators

If you want to controll all agents locally, you dont need a sim state server running. Below we create and example
Scenario where we spawn the ego in random locations and have random objects appear in front of it

```python
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.transformation import Transformation
import paralleldomain.data_lab as data_lab
import dataclasses

# only a front facing camera this time
sensor_rig = data_lab.SensorRig().add_camera(
    name="Front",
    width=768,
    height=768,
    frame_rate=4.0,
    field_of_view_degrees=70,
    pose=Transformation.from_euler_angles(
        angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
    ),
    annotation_types=[AnnotationTypes.SemanticSegmentation2D],
)

# same environment as before
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Night, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.1)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.0.1"))


# We create a agent behaviour that spawns at a random map location at each time step
class RandomStreetPoseBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        relative_location_variance: float,
        direction_variance_in_degrees: float,
    ):
        super().__init__()
        self._initial_pose: Transformation = None
        self.location_seed = data_lab.RandomVariable[int](default=(0, 100000))
        self.relative_location_variance = relative_location_variance
        self.direction_variance_in_degrees = direction_variance_in_degrees

    def set_inital_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        pose = sim_state.map.get_random_street_location(
            relative_location_variance=self.relative_location_variance,
            direction_variance_in_degrees=self.direction_variance_in_degrees,
            random_seed=self.location_seed.sample(),
        )
        self._initial_pose = pose
        agent.set_pose(pose=pose)

    def update_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        self.set_inital_state(sim_state=sim_state, agent=agent)


# Now we create a vehicle agent with a random car asset and add the custom ego behavior to it
scenario.add_ego(
    data_lab.SimAgents.create_ego_vehicle(sensor_rig=sensor_rig).set_behaviour(
        RandomStreetPoseBehaviour(
            relative_location_variance=0.1,
            direction_variance_in_degrees=0.0,
        )
    )
)


# Now lets create a generator that places objects in front of the ego in each frame
class BlockEgoBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        vertical_offset: data_lab.RandomVariable[float],
        dist_to_ego: data_lab.RandomVariable[float],
        yaw_angle: data_lab.RandomVariable[float],
    ):
        super().__init__()
        self.dist_to_ego = dist_to_ego
        self.vertical_offset = vertical_offset
        self.yaw_angle = yaw_angle

    def set_inital_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        pos_in_ego_coords = data_lab.SimCoordinateSystem.get_local_forward_direction() * self.dist_to_ego.sample()
        vert_offset = data_lab.SimCoordinateSystem.get_local_left_direction() * self.vertical_offset.sample()
        pos_in_ego_coords += vert_offset
        # We always access the current frame ego post not the previous frame to make sure we know
        # the new position of the ego
        pose_in_front_of_ego = sim_state.ego_pose @ pos_in_ego_coords
        pose = Transformation.from_euler_angles(
            translation=pose_in_front_of_ego, degrees=True, order="xyz", angles=[0.0, 0.0, self.yaw_angle.sample()]
        )
        agent.set_pose(pose=pose)

    def update_state(self, sim_state: data_lab.SimState, agent: data_lab.CustomSimulationAgent):
        self.set_inital_state(sim_state=sim_state, agent=agent)


@dataclasses.dataclass
class EgoBlockObjectsGenerator(data_lab.CustomAtomicGenerator):
    number_of_agents: data_lab.RandomVariable[int] = data_lab.RandomVariable(default=1)
    distance_to_ego: data_lab.RandomVariable[float] = data_lab.RandomVariable(default=10.0)
    vertical_offset: data_lab.RandomVariable[float] = data_lab.RandomVariable(default=(-0.2, 0.2))
    yaw_angle: data_lab.RandomVariable[float] = data_lab.RandomVariable(default=(-180.0, 180.0))
    objects_to_place: data_lab.RandomVariable[str] = data_lab.RandomVariable(
        default=["character", "vehicle", "traffic_control", "prop"]
    )

    def create_agents_for_new_scene(self, state: data_lab.SimState):
        agents = []
        for _ in range(int(self.number_of_agents.sample())):
            asset_category = self.objects_to_place.sample()
            if asset_category == "vehicle":
                agent = data_lab.SimAgents.create_vehicle()
            elif asset_category == "character":
                agent = data_lab.SimAgents.create_pedestrian()
            else:
                agent = data_lab.SimAgents.create_object(asset_category=asset_category)
            agent = agent.set_behaviour(
                BlockEgoBehaviour(
                    dist_to_ego=self.distance_to_ego.clone(),
                    vertical_offset=self.vertical_offset.clone(),
                    yaw_angle=self.yaw_angle.clone(),
                )
            )
            agents.append(agent)
        return agents


# Now we add our custom gen and set the distributions we want
custom_gen = EgoBlockObjectsGenerator()
custom_gen.distance_to_ego.set_uniform_distribution(3, 50)
custom_gen.number_of_agents.set_uniform_distribution(15, 50)
custom_gen.vertical_offset.set_uniform_distribution(-20, 20)
scenario.add_agents(custom_gen)

# Now lets preview our custom locally simulated scenario
data_lab.preview_scenario(
    scenario=scenario,
    dataset_name="test",
    number_of_scenes=1,
    frames_per_scene=5,  # controls the render quality. Higher is better, but slower
)
```
