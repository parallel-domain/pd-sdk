from tempfile import TemporaryDirectory
from typing import Optional
from unittest import mock

import numpy as np
import pd.state
import pytest
from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.config.environment import TimeOfDays
from pd.data_lab.config.location import Location
from pd.data_lab.generators.non_atomics import NonAtomicGeneratorMessage
from pd.data_lab.generators.simulation_agent import SimulationAgentBase
from pd.data_lab.render_instance import AbstractRenderInstance
from pd.data_lab.scenario import Scenario
from pd.data_lab.sim_instance import AbstractSimulationInstance
from pd.management import Ig
from pd.session import SimSession, StepSession

from paralleldomain.data_lab import (
    CustomSimulationAgent,
    CustomSimulationAgentBehaviour,
    CustomSimulationAgents,
    ExtendedSimState,
    create_frame_stream,
    encode_sim_states,
)
from paralleldomain.data_lab.config.sensor_rig import SensorConfig, SensorRig
from paralleldomain.data_lab.generators.debris import DebrisGeneratorParameters
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
    SpecialAgentTag,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation
from test_paralleldomain.data_lab.constants import LOCATIONS


class MockRenderInstance(AbstractRenderInstance):
    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario
        self.session = mock.MagicMock()
        self.session.query_sensor_data = self.query_sensor_data
        self.session.update_state = self.update_state
        self._mock_step_ig = mock.MagicMock()
        self._last_timestamp = -1.0

    def query_sensor_data(self, agent_id, sensor_name, buffer_type):
        sensor = next(iter([s for s in self.scenario.sensor_rig.sensor_configs if s.display_name == sensor_name]))
        width = sensor.camera_intrinsic.width
        height = sensor.camera_intrinsic.height
        if pd.state.SensorBuffer.SEGMENTATION == buffer_type:
            data = (np.ones((height, width, 2), dtype=int) * 255).astype(np.uint8)
        elif pd.state.SensorBuffer.INSTANCES == buffer_type:
            data = (np.ones((height, width, 2), dtype=int) * 255).astype(np.uint8)
        elif pd.state.SensorBuffer.RGB == buffer_type:
            data = (np.ones((height, width, 3), dtype=int) * 255).astype(np.uint8)
        elif pd.state.SensorBuffer.NORMALS == buffer_type:
            data = (np.ones((height, width, 3), dtype=int) * 255).astype(np.uint8)
        else:
            raise NotImplementedError()
        sensor_data = pd.state.SensorData(data=data, width=width, height=height)
        return sensor_data

    def update_state(self, state: pd.state.State):
        assert state.simulation_time_sec > self._last_timestamp, "Timestamps sent to the ig need to increase"
        self._last_timestamp = float(state.simulation_time_sec)

    def __enter__(self) -> StepSession:
        self._last_timestamp = -1.0
        self._simulation_time = 0.0
        return self.session

    def __exit__(self):
        pass

    @property
    def step_ig(self) -> Optional[Ig]:
        return self._mock_step_ig

    @property
    def loaded_location(self) -> Optional[Location]:
        return self.scenario.location


class MockSimulationInstance(AbstractSimulationInstance):
    def __init__(self, scenario: Scenario, ego_agent: Optional[SimulationAgentBase]):
        super().__init__()
        self.scenario = scenario
        self._ego_agent = ego_agent
        self.query_sim_state_calls = 0

    def query_sim_state(self) -> pd.state.State:
        self._simulation_time_sec += self.scenario.sim_state.scenario_gen.sim_update_time
        sim_state = pd.state.State(
            simulation_time_sec=self._simulation_time_sec,
            world_info=pd.state.WorldInfo(),
            agents=[self.ego_agent.step_agent],
        )
        self.query_sim_state_calls += 1
        return sim_state

    def __enter__(self) -> SimSession:
        ego_agent = self._ego_agent
        if ego_agent is None:
            ego_agent = mock.MagicMock()
            ego_agent.agent_id = 42
            ego_agent.step_agent = pd.state.ModelAgent(id=42, asset_name="", pose=np.eye(4), velocity=(0.0, 0.0, 0.0))
        self.ego_agent = ego_agent
        self._session = mock.MagicMock()
        self._session.load_scenario_generation.return_value = (self.scenario.location, ego_agent.agent_id)
        self._simulation_time_sec = 0.0
        return self._session

    def __exit__(self):
        pass

    @property
    def session(self) -> SimSession:
        return self._session


class TestScenario:
    @pytest.fixture(params=list(LOCATIONS.keys()))
    def location(self, request):
        map_name = request.param
        location = LOCATIONS[map_name]
        return location

    @pytest.fixture
    def atomic_only_scenario(self, location: Location) -> Scenario:
        sensor_rig = SensorRig(
            sensor_configs=[
                SensorConfig.create_camera_sensor(
                    name="Front",
                    width=1920,
                    height=1080,
                    field_of_view_degrees=70,
                    pose=Transformation.from_euler_angles(
                        angles=[2.123, 3.0, -180.0], order="xyz", degrees=True, translation=[10.1, -12.0, 2.0]
                    ),
                    annotation_types=[
                        AnnotationTypes.SurfaceNormals2D,
                        AnnotationTypes.SemanticSegmentation2D,
                        AnnotationTypes.InstanceSegmentation2D,
                    ],
                )
            ]
        )

        scenario = Scenario(sensor_rig=sensor_rig)
        scenario.set_location(location)
        scenario.environment.time_of_day.set_category_weight(TimeOfDays.Day, 1.0)
        scenario.environment.time_of_day.set_category_weight(TimeOfDays.Dawn, 1.0)
        scenario.environment.time_of_day.set_category_weight(TimeOfDays.Dusk, 1.0)
        scenario.environment.time_of_day.set_category_weight(TimeOfDays.Night, 1.0)
        # Place other agents
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=[SpecialAgentTag.EGO],
                        max_spawn_radius=200.0,
                    )
                ),
            )
        )

        return scenario

    @pytest.fixture
    def atomic_only_scenario_storage_path(self, atomic_only_scenario: Scenario) -> AnyPath:
        with TemporaryDirectory() as tmp_dir:
            tmp_file = AnyPath(tmp_dir) / "test.json"
            atomic_only_scenario.save_scenario(path=tmp_file)
            yield tmp_file

    def check_atomic_only_scenario(self, scenario: Scenario, target_locatio: Location):
        assert scenario.sensor_rig is not None
        assert len(scenario.sensor_rig.sensors) == 1
        assert len(scenario.sensor_rig.cameras) == 1
        assert len(scenario.sensor_rig.camera_names) == 1
        assert "Front" in scenario.sensor_rig.camera_names

        camera = scenario.sensor_rig.cameras[0]
        assert camera.is_camera
        assert camera.camera_intrinsic.width == 1920
        assert camera.camera_intrinsic.height == 1080
        assert camera.camera_intrinsic.fov == 70.0
        assert AnnotationTypes.SurfaceNormals2D in camera.annotations_types
        assert AnnotationTypes.SemanticSegmentation2D in camera.annotations_types
        assert AnnotationTypes.InstanceSegmentation2D in camera.annotations_types
        extrinsic = camera.sensor_to_ego
        roll, pitch, yaw = extrinsic.as_euler_angles(order="xyz", degrees=True)
        assert np.allclose(roll, 2.123)
        assert np.allclose(pitch, 3.0)
        assert np.allclose(yaw, -180.0)
        x, y, z = extrinsic.translation
        assert np.allclose(x, 10.1)
        assert np.allclose(y, -12.0)
        assert np.allclose(z, 2.0)

        assert target_locatio.name == scenario.location.name
        # assert "v2.0.1" == scenario.location.version

        time_of_day_distribution = scenario.environment.time_of_day
        assert len(time_of_day_distribution.buckets) == 4
        tod = [b.value for b in time_of_day_distribution.buckets]
        assert all([b.probability == 1.0 for b in time_of_day_distribution.buckets])
        assert "DAWN" in tod
        assert "DAY" in tod
        assert "DUSK" in tod
        assert "NIGHT" in tod

        assert len(scenario.pd_generators) == 1
        assert not isinstance(scenario.pd_generators[0], NonAtomicGeneratorMessage)

    def test_load_from_json_with_sensor_rig(self, atomic_only_scenario_storage_path: AnyPath, location: Location):
        scenario = Scenario.load_scenario(path=atomic_only_scenario_storage_path)
        self.check_atomic_only_scenario(scenario=scenario, target_locatio=location)

    def test_save_and_load_from_json_with_sensor_rig(
        self, atomic_only_scenario_storage_path: AnyPath, location: Location
    ):
        scenario = Scenario.load_scenario(path=atomic_only_scenario_storage_path)

        with TemporaryDirectory() as tmp_dir:
            tmp_file = AnyPath(tmp_dir) / "test.json"
            scenario.save_scenario(path=tmp_file)

            new_reload_scenario = Scenario.load_scenario(path=tmp_file)
            self.check_atomic_only_scenario(scenario=new_reload_scenario, target_locatio=location)

    def test_add_atomics(self, atomic_only_scenario: Scenario):
        scenario = atomic_only_scenario

        assert len(scenario.pd_generators) == 1
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

        assert len(scenario.pd_generators) == 2

        scenario.add_objects(
            generator=DebrisGeneratorParameters(
                max_debris_distance=25.0,
                spawn_probability=0.7,
                debris_asset_tag="trash_bottle_tall_01",
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=[SpecialAgentTag.EGO],
                    )
                ),
            )
        )

        assert len(scenario.pd_generators) == 3

        # Place other agents
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=[SpecialAgentTag.EGO],
                        max_spawn_radius=200.0,
                    )
                ),
            )
        )

        assert len(scenario.pd_generators) == 4

        cloned = scenario.clone()

        assert len(cloned.pd_generators) == 4

    def test_save_load_and_add_atomics(self, atomic_only_scenario: Scenario):
        scenario = atomic_only_scenario
        # Place other agents
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=[SpecialAgentTag.EGO],
                        max_spawn_radius=200.0,
                    )
                ),
            )
        )

        assert len(scenario.pd_generators) == 2

        with TemporaryDirectory() as tmp_dir:
            tmp_file = AnyPath(tmp_dir) / "test.json"
            tmp_file = str(tmp_file.absolute())
            scenario.save_scenario(path=tmp_file)

            new_reload_scenario = Scenario.load_scenario(path=tmp_file)

            assert len(new_reload_scenario.pd_generators) == 2
            new_reload_scenario.add_objects(
                generator=DebrisGeneratorParameters(
                    max_debris_distance=25.0,
                    spawn_probability=0.7,
                    debris_asset_tag="trash_bottle_tall_01",
                    position_request=PositionRequest(
                        location_relative_position_request=LocationRelativePositionRequest(
                            agent_tags=[SpecialAgentTag.EGO],
                        )
                    ),
                )
            )

            assert len(new_reload_scenario.pd_generators) == 3
            cloned = new_reload_scenario.clone()
            assert len(cloned.pd_generators) == 3

    def run_mocked_frame_generation(
        self,
        scenario: Scenario,
        ego_agent: SimulationAgentBase = None,
        frames_per_scene: int = 10,
        number_of_scenes: int = 1,
    ):
        sensor_rig = scenario.sensor_rig
        render_instance = MockRenderInstance(scenario=scenario)
        sim_instance = MockSimulationInstance(scenario=scenario, ego_agent=ego_agent)

        frame_count = 0
        for frame, scene in create_frame_stream(
            scenario=scenario,
            frames_per_scene=frames_per_scene,
            number_of_scenes=number_of_scenes,
            sim_instance=sim_instance,
            render_instance=render_instance,
            dataset_name="test",
        ):
            assert len(frame.camera_names) == len(sensor_rig.cameras)
            for camera_frame in frame.camera_frames:
                img = camera_frame.image.rgb
                sensor = next(
                    iter([s for s in scenario.sensor_rig.sensor_configs if s.display_name == camera_frame.sensor_name])
                )
                width = sensor.camera_intrinsic.width
                height = sensor.camera_intrinsic.height
                assert img.shape == (height, width, 3)
                frame_count += 1
        assert frame_count == (frames_per_scene * len(sensor_rig.cameras)) * number_of_scenes

    def test_scenario_custom(self):
        cnt_mock = mock.MagicMock()
        cnt_mock.setup_count = 0
        cnt_mock.update_state_count = 0
        cnt_mock.clone_count = 0

        class TestBehaviour(CustomSimulationAgentBehaviour):
            def __init__(self, counter: mock.MagicMock):
                self.counter = counter

            def set_inital_state(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, random_seed: int):
                self.counter.setup_count += 1

            def update_state(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent):
                self.counter.update_state_count += 1

            def clone(self) -> "TestBehaviour":
                self.counter.clone_count += 1
                return TestBehaviour(counter=self.counter)

        sensor_rig = SensorRig().add_camera(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
        )

        scenario = Scenario(sensor_rig=sensor_rig)
        scenario.set_location(Location(name="SF_6thAndMission_medium", version="v2.0.1"))
        ego_agent = CustomSimulationAgents.create_ego_vehicle(sensor_rig=sensor_rig).set_behaviour(
            TestBehaviour(counter=cnt_mock)
        )
        scenario.add_ego(ego_agent)
        frames_per_scene = 100
        update_calls = frames_per_scene * scenario.sim_state.scenario_gen.sim_capture_rate
        self.run_mocked_frame_generation(scenario=scenario, ego_agent=ego_agent, frames_per_scene=frames_per_scene)
        assert cnt_mock.setup_count == 1
        assert cnt_mock.update_state_count == update_calls
        assert cnt_mock.clone_count == 1

    def test_scenario_custom_and_atomic(self):
        cnt_mock = mock.MagicMock()
        cnt_mock.setup_count = 0
        cnt_mock.update_state_count = 0
        cnt_mock.clone_count = 0

        class TestBehaviour(CustomSimulationAgentBehaviour):
            def __init__(self, counter: mock.MagicMock):
                self.counter = counter

            def set_inital_state(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent, random_seed: int):
                self.counter.setup_count += 1

            def update_state(self, sim_state: ExtendedSimState, agent: CustomSimulationAgent):
                self.counter.update_state_count += 1

            def clone(self) -> "TestBehaviour":
                self.counter.clone_count += 1
                return TestBehaviour(counter=self.counter)

        sensor_rig = SensorRig().add_camera(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
        )

        scenario = Scenario(sensor_rig=sensor_rig)
        scenario.set_location(Location(name="SF_6thAndMission_medium", version="v2.0.1"))
        ego_agent = CustomSimulationAgents.create_ego_vehicle(sensor_rig=sensor_rig).set_behaviour(
            TestBehaviour(counter=cnt_mock)
        )
        scenario.add_agents(ego_agent)

        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=[SpecialAgentTag.EGO],
                        max_spawn_radius=200.0,
                    )
                ),
            )
        )

        frames_per_scene = 100
        # since we run start_skip_frames the pd gens are already called 9 times and hence we only call the custom
        # gen once for the first render
        update_calls = 1 + (frames_per_scene - 1) * scenario.sim_state.scenario_gen.sim_capture_rate
        self.run_mocked_frame_generation(scenario=scenario, ego_agent=ego_agent, frames_per_scene=frames_per_scene)
        assert cnt_mock.setup_count == 1
        assert cnt_mock.update_state_count == update_calls
        assert cnt_mock.clone_count == 1

    def test_scenario_atomics_in_loop(self, atomic_only_scenario: Scenario):
        for _ in range(3):
            self.run_mocked_frame_generation(scenario=atomic_only_scenario)

    def test_scenario_atomics_multiple_scenes(self, atomic_only_scenario: Scenario):
        self.run_mocked_frame_generation(scenario=atomic_only_scenario, number_of_scenes=3, frames_per_scene=10)

    @pytest.mark.parametrize(
        "yield_every_sim_state",
        [(True,), (False,)],
    )
    def test_sim_state_encode(self, yield_every_sim_state: bool, atomic_only_scenario: Scenario):
        sim_instance = MockSimulationInstance(scenario=atomic_only_scenario, ego_agent=None)

        number_of_scenes = 4
        frames_per_scene = 20

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = AnyPath(tmp_dir)

            encode_sim_states(
                scenario=atomic_only_scenario,
                output_folder=tmp_dir,
                number_of_scenes=number_of_scenes,
                frames_per_scene=frames_per_scene,
                sim_instance=sim_instance,
                render_instance=None,
                yield_every_sim_state=yield_every_sim_state,
            )

            if yield_every_sim_state:
                every_frame_and_warmup_count = (
                    number_of_scenes * frames_per_scene * atomic_only_scenario.sim_state.scenario_gen.sim_capture_rate
                    + number_of_scenes * atomic_only_scenario.sim_state.scenario_gen.start_skip_frames
                )
                assert sim_instance.query_sim_state_calls == every_frame_and_warmup_count
            else:
                # only capture frames are yielded
                assert (
                    sim_instance.query_sim_state_calls
                    == number_of_scenes
                    * (frames_per_scene - 1)
                    * atomic_only_scenario.sim_state.scenario_gen.sim_capture_rate
                    + number_of_scenes * atomic_only_scenario.sim_state.scenario_gen.start_skip_frames
                    + number_of_scenes
                )

            assert len(list(tmp_dir.iterdir())) == number_of_scenes
            for dir in tmp_dir.iterdir():
                if yield_every_sim_state:
                    assert (
                        len(list(dir.iterdir()))
                        == frames_per_scene * atomic_only_scenario.sim_state.scenario_gen.sim_capture_rate
                    )
                else:
                    assert len(list(dir.iterdir())) == frames_per_scene
                for file in dir.iterdir():
                    assert file.name.endswith(".pd")
                    decoded = pd.state.bytes_to_state(file.open("rb").read())
                    assert isinstance(decoded, pd.state.State)
                    # in our mock we just add 1 agent
                    assert len(decoded.agents) == 1
