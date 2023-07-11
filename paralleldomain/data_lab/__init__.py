import logging
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar, Union

import pd.data_lab.config.environment as _env
import pd.data_lab.config.location as _loc
import pypeln
from pd.data_lab import TSimState, create_sensor_sim_stream, encode_sim_states
from pd.data_lab.generators.custom_generator import CustomAtomicGenerator
from pd.data_lab.generators.custom_simulation_agent import (
    CustomSimulationAgent,
    CustomSimulationAgentBehaviour,
    CustomSimulationAgents,
)
from pd.data_lab.scenario import Scenario, ScenarioSource, SimulatedScenarioCollection
from pd.data_lab.sim_instance import FromDiskSimulation
from pd.data_lab.state_callback import StateCallback
from pd.state import ModelAgent, VehicleAgent

from paralleldomain import Scene
from paralleldomain.data_lab.config.reactor import ReactorConfig
from paralleldomain.data_lab.config.sensor_rig import SensorConfig, SensorRig
from paralleldomain.data_lab.reactor import (
    ReactorFrameStreamGenerator,
    ReactorInputLoader,
    clear_output_path,
    get_instance_mask_and_prompts,
    get_mask_annotations,
    instance_inpainting,
    merge_instance_masks,
)
from paralleldomain.data_lab.sim_state import ExtendedSimState
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.step.frame_decoder import StepFrameDecoder
from paralleldomain.decoding.step.scene_decoder import StepSceneDecoder
from paralleldomain.encoding.helper import get_encoding_format
from paralleldomain.encoding.stream_pipeline_builder import StreamDatasetPipelineEncoder, StreamEncodingPipelineBuilder
from paralleldomain.encoding.stream_pipeline_item import StreamPipelineItem
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.frame import Frame
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.visualization.model_visualization import show_frame
from paralleldomain.visualization.state_visualization import show_agents, show_map

TimeOfDays = _env.TimeOfDays
Location = _loc.Location
CustomSimulationAgentBehaviour = CustomSimulationAgentBehaviour[ExtendedSimState]
CustomSimulationAgents = CustomSimulationAgents[ExtendedSimState]
CustomSimulationAgent = CustomSimulationAgent[ExtendedSimState]
CustomAtomicGenerator = CustomAtomicGenerator[ExtendedSimState]
encode_sim_states = encode_sim_states
_ = Scenario


# TODO: LFU to FLU (left to right hand problem)
# coordinate_system = INTERNAL_COORDINATE_SYSTEM
coordinate_system = CoordinateSystem("RFU")

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])

logger = logging.getLogger(__name__)


class FilterAsset(StateCallback):
    def __init__(self, asset_name: str):
        self._asset_name = asset_name

    def __call__(self, sim_state: TSimState):
        for agent in sim_state.current_agents:
            if agent.agent_id == sim_state.ego_agent_id:
                continue
            if isinstance(agent.step_agent, ModelAgent) and agent.step_agent.asset_name == self._asset_name:
                sim_state.remove_agent(agent=agent)
            elif isinstance(agent.step_agent, VehicleAgent) and agent.step_agent.vehicle_type == self._asset_name:
                sim_state.remove_agent(agent=agent)

    def clone(self) -> "StateCallback":
        return FilterAsset(asset_name=self._asset_name)


class SimStateVisualizerCallback(StateCallback):
    def __init__(self):
        self._logged_map = False

    def __call__(self, sim_state: ExtendedSimState):
        if not self._logged_map:
            show_map(umd_map=sim_state.map)
            self._logged_map = True

        show_agents(sim_state=sim_state.current_state)

    def clone(self) -> "StateCallback":
        return SimStateVisualizerCallback()


def _create_decoded_stream_from_scenario(
    scenario: ScenarioSource,
    scenario_index: int,
    dataset_name: str = "Default Dataset Name",
    end_skip_frames: Optional[int] = None,
    estimated_startup_time: int = 180,
    frames_per_scene: Optional[int] = None,
    seconds_per_frame_estimate: int = 100,
    sim_capture_rate: Optional[int] = None,
    sim_settle_frames: Optional[int] = None,
    start_skip_frames: Optional[int] = None,
    use_merge_batches: Optional[bool] = None,
    **kwargs,
) -> Generator[Tuple[Frame[TDateTime], Scene], None, None]:
    discrete_scenario, gen = create_sensor_sim_stream(
        scenario=scenario,
        sim_settle_frames=sim_settle_frames,
        scenario_index=scenario_index,
        dataset_name=dataset_name,
        frames_per_scene=frames_per_scene,
        estimated_startup_time=estimated_startup_time,
        end_skip_frames=end_skip_frames,
        start_skip_frames=start_skip_frames,
        use_merge_batches=use_merge_batches,
        sim_capture_rate=sim_capture_rate,
        seconds_per_frame_estimate=seconds_per_frame_estimate,
        sim_state_type=ExtendedSimState,
        **kwargs,
    )
    scene_name = discrete_scenario.name
    scene = None
    og_scene = None
    for temporal_sensor_session_reference, sim_state in gen:
        if scene is None:
            decoder = StepSceneDecoder(
                sensor_rig=sim_state.sensor_rig,
                dataset_name=dataset_name,
                settings=DecoderSettings(),
            )

            scene = Scene(
                decoder=decoder,
                available_annotation_types=decoder.available_annotations,
                name=scene_name,
            )
            og_scene = scene
            # cache all data in memory so pickling is possible
            scene_decoder = InMemorySceneDecoder.from_scene(scene=scene)
            scene = Scene(
                decoder=scene_decoder,
                available_annotation_types=scene.available_annotation_types,
                name=scene_name,
            )
        current_frame = Frame[datetime](
            frame_id=sim_state.current_frame_id,
            decoder=StepFrameDecoder(
                dataset_name=dataset_name,
                date_time=sim_state.current_sim_date_time,
                ego_agent_id=sim_state.ego_agent_id,
                scene_name=scene_name,
                sensor_rig=sim_state.sensor_rig,
                session=temporal_sensor_session_reference,
                settings=DecoderSettings(),
            ),
        )
        scene_decoder.frame_ids.append(sim_state.current_frame_id)
        scene_decoder.frame_id_to_date_time_map[sim_state.current_frame_id] = sim_state.current_sim_date_time
        # cache all data in memory so that the sim and renderer can update
        in_memory_frame_decoder = InMemoryFrameDecoder.from_frame(frame=current_frame)
        yield Frame[TDateTime](
            frame_id=current_frame.frame_id,
            decoder=in_memory_frame_decoder,
        ), scene
    if og_scene is not None:
        og_scene.clear_from_cache()


def create_frame_stream(
    scenario: ScenarioSource,
    number_of_scenes: int = 1,
    frames_per_scene: Optional[int] = None,
    number_of_instances: int = 1,
    instance_run_env: str = "sync",
    statistic: Statistic = None,
    **kwargs,
) -> Generator[Tuple[Frame[TDateTime], Scene], None, None]:
    if number_of_instances == 1 and instance_run_env == "sync":
        for scenario_index in range(number_of_scenes):
            for frame, scene in _create_decoded_stream_from_scenario(
                scenario=scenario, scenario_index=scenario_index, frames_per_scene=frames_per_scene, **kwargs
            ):
                if statistic is not None:
                    for sensor_frame in frame.sensor_frames:
                        statistic.update(scene=scene, sensor_frame=sensor_frame)
                yield frame, scene
    else:
        if instance_run_env == "thread":
            run_env = pypeln.thread
        elif instance_run_env == "process":
            run_env = pypeln.process
        else:
            run_env = pypeln.sync

        def _endless_stream():
            scene_index = 0
            while True:
                yield scene_index
                scene_index += 1

        scene_index_iter = range(number_of_scenes) if number_of_scenes >= 0 else _endless_stream()

        streams = pypeln.sync.from_iterable(scene_index_iter)
        streams = run_env.flat_map(
            lambda i: _create_decoded_stream_from_scenario(
                scenario=scenario, scenario_index=i, frames_per_scene=frames_per_scene, **kwargs
            ),
            streams,
            workers=number_of_instances,
            maxsize=number_of_instances,
        )
        for frame, scene in pypeln.sync.to_iterable(streams, maxsize=2 * number_of_instances):
            if statistic is not None:
                for sensor_frame in frame.sensor_frames:
                    statistic.update(scene=scene, sensor_frame=sensor_frame)
            yield frame, scene


def create_reactor_frame_stream(
    scenario: ScenarioSource,
    reactor_config: ReactorConfig,
    format_kwargs: Dict[str, Any],
    use_cached_reactor_states: bool = False,
    instance_run_env: str = "sync",
    **kwargs,
) -> Generator[Tuple[Frame[TDateTime], Scene], None, None]:
    dataset_output_path = format_kwargs["dataset_output_path"]
    simstate_dir = AnyPath(dataset_output_path + "/cache/sim_states")
    encode_dir = AnyPath(dataset_output_path + "/cache/dataset")

    if use_cached_reactor_states is True and simstate_dir.exists():
        collection = SimulatedScenarioCollection(storage_folder=simstate_dir)
    else:
        copy_kwargs = kwargs.copy()
        copy_kwargs.pop("render_instance", None)
        encode_sim_states(
            scenario=scenario,
            output_folder=simstate_dir,
            render_instance=None,
            sim_state_type=ExtendedSimState,
            **copy_kwargs,
        )
        collection = SimulatedScenarioCollection(storage_folder=simstate_dir)

    if use_cached_reactor_states is True and encode_dir.exists():
        clear_output_path(output_path=AnyPath(dataset_output_path))
        stored_dataset = decode_dataset(dataset_path=encode_dir, dataset_format="dgpv1")
    else:
        copy_kwargs = kwargs.copy()
        copy_kwargs.pop("sim_instance", None)
        logger.info("Reactor: creating auxiliary dataset.")
        create_mini_batch(
            scenario=collection,
            format_kwargs=dict(
                dataset_output_path=encode_dir,
                encode_to_binary=False,
            ),
            sim_instance=FromDiskSimulation(),
            pipeline_kwargs=dict(copy_all_available_sensors_and_annotations=True, run_env="thread"),
            reactor_config=None,
            **copy_kwargs,
        )
        stored_dataset = decode_dataset(dataset_path=encode_dir, dataset_format="dgpv1")

    filtered_collection = SimulatedScenarioCollection(storage_folder=simstate_dir)
    filtered_collection.add_state_callback(
        state_callback=FilterAsset(asset_name=reactor_config.reactor_object.asset_name)
    )

    copy_kwargs = kwargs.copy()
    copy_kwargs.pop("sim_instance", None)
    reactor_input_loader = ReactorInputLoader(reactor_config=reactor_config, stored_dataset=stored_dataset)
    reactor_frame_stream_generator = ReactorFrameStreamGenerator(reactor_config=reactor_config)

    logger.info("Reactor: creating reactor dataset.")
    if instance_run_env == "thread":
        run_env = pypeln.thread
    elif instance_run_env == "process":
        run_env = pypeln.process
    else:
        run_env = pypeln.sync

    yield from (
        create_frame_stream(
            scenario=filtered_collection,
            sim_instance=FromDiskSimulation(),
            instance_run_env=instance_run_env,
            **copy_kwargs,
        )
        | run_env.map(f=reactor_input_loader.load_reactor_input, workers=1, maxsize=1)
        | run_env.map(f=reactor_frame_stream_generator.create_reactor_frame, workers=1, maxsize=1)
    )


def _get_encoder(
    scenario: ScenarioSource,
    number_of_scenes: int,
    frames_per_scene: int,
    number_of_instances: int = 1,
    instance_run_env: str = "thread",
    format: str = "dgpv1",
    use_tqdm: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: Statistic = None,
    **kwargs,
) -> StreamDatasetPipelineEncoder:
    format_kwargs = format_kwargs if format_kwargs is not None else dict()
    pipeline_kwargs = pipeline_kwargs if pipeline_kwargs is not None else dict()

    if number_of_scenes < 1:
        raise ValueError("A number of scenes > 0 has to be passed!")

    if frames_per_scene < 1:
        raise ValueError("A number of frames per scene > 0 has to be passed!")

    encoding_format = get_encoding_format(format_name=format, **format_kwargs)
    if reactor_config is not None:
        frame_stream = create_reactor_frame_stream(
            scenario=scenario,
            number_of_scenes=number_of_scenes,
            frames_per_scene=frames_per_scene,
            reactor_config=reactor_config,
            format_kwargs=format_kwargs,
            use_cached_reactor_states=use_cached_reactor_states,
            instance_run_env=instance_run_env,
            **kwargs,
        )
    else:
        frame_stream = create_frame_stream(
            scenario=scenario,
            number_of_scenes=number_of_scenes,
            frames_per_scene=frames_per_scene,
            number_of_instances=number_of_instances,
            instance_run_env=instance_run_env,
            statistic=statistic,
            **kwargs,
        )
    pipeline_builder = StreamEncodingPipelineBuilder(
        frame_stream=frame_stream,
        number_of_scenes=number_of_scenes,
        number_of_frames_per_scene=frames_per_scene,
        **pipeline_kwargs,
    )
    encoder = StreamDatasetPipelineEncoder.from_builder(
        use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format
    )
    return encoder


def create_mini_batch(
    scenario: ScenarioSource,
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    number_of_instances: int = 1,
    instance_run_env: str = "thread",
    scene_name_gen: Callable[[int], str] = None,
    format: str = "dgpv1",
    use_tqdm: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: Statistic = None,
    **kwargs,
):
    encoder = _get_encoder(
        scenario=scenario,
        number_of_scenes=number_of_scenes,
        frames_per_scene=frames_per_scene,
        scene_name_gen=scene_name_gen,
        format=format,
        number_of_instances=number_of_instances,
        instance_run_env=instance_run_env,
        use_tqdm=use_tqdm,
        reactor_config=reactor_config,
        use_cached_reactor_states=use_cached_reactor_states,
        format_kwargs=format_kwargs,
        pipeline_kwargs=pipeline_kwargs,
        statistic=statistic,
        **kwargs,
    )
    encoder.encode_dataset()


def create_mini_batch_stream(
    scenario: ScenarioSource,
    format: str = "dgpv1",
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    number_of_instances: int = 1,
    instance_run_env: str = "thread",
    scene_name_gen: Callable[[int], str] = None,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: Statistic = None,
    **kwargs,
) -> Generator[StreamPipelineItem, None, None]:
    encoder = _get_encoder(
        scenario=scenario,
        format=format,
        number_of_instances=number_of_instances,
        instance_run_env=instance_run_env,
        number_of_scenes=number_of_scenes,
        frames_per_scene=frames_per_scene,
        scene_name_gen=scene_name_gen,
        use_tqdm=False,
        format_kwargs=format_kwargs,
        pipeline_kwargs=pipeline_kwargs,
        statistic=statistic,
        **kwargs,
    )
    if statistic is not None:
        for item in encoder.yielding_encode_dataset():
            statistic.update(scene=item.scene, sensor_frame=item.sensor_frame)
            yield item
    else:
        yield from encoder.yielding_encode_dataset()


def preview_scenario(
    scenario: ScenarioSource,
    number_of_scenes: int = 1,
    frames_per_scene: int = 10,
    annotations_to_show: List[AnnotationType] = None,
    statistic: Statistic = None,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    **kwargs,
):
    if not any([isinstance(cb, SimStateVisualizerCallback) for cb in scenario.state_callbacks]):
        scenario.add_state_callback(SimStateVisualizerCallback())

    if kwargs.get("render_instance", None) is None:
        for scenario_index in range(number_of_scenes):
            _, gen = create_sensor_sim_stream(
                scenario=scenario,
                scenario_index=scenario_index,
                frames_per_scene=frames_per_scene,
                sim_state_type=ExtendedSimState,
                **kwargs,
            )
            for _ in gen:
                pass
    elif reactor_config is not None:
        for frame, scene in create_reactor_frame_stream(
            scenario=scenario,
            number_of_scenes=number_of_scenes,
            frames_per_scene=frames_per_scene,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)
    else:
        for frame, _ in create_frame_stream(
            scenario=scenario,
            frames_per_scene=frames_per_scene,
            number_of_scenes=number_of_scenes,
            statistic=statistic,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)
