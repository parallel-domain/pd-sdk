import logging
import os
import tempfile
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, overload

import pd.data_lab.config.environment as _env
import pd.data_lab.config.location as _loc
import pd.management
import py7zr
import pypeln
from pd.core import PdError
from pd.data_lab import LabeledStateReference, TSimState, create_sensor_sim_stream, encode_sim_states
from pd.data_lab.generators.custom_generator import CustomAtomicGenerator
from pd.data_lab.generators.custom_simulation_agent import (
    CustomSimulationAgent,
    CustomSimulationAgentBehavior,
    CustomSimulationAgents,
)
from pd.data_lab.label_engine_instance import LabelEngineInstance
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Scenario, ScenarioCreator, SimulatedScenarioCollection, scene_index_to_name
from pd.data_lab.sim_instance import FromDiskSimulation, SimulationInstance, SimulationStateProvider
from pd.data_lab.state_callback import StateCallback
from pd.label_engine import DEFAULT_LABEL_ENGINE_CONFIG_NAME
from pd.management import Ig
from pd.state import ModelAgent, VehicleAgent, state_to_bytes

from paralleldomain import Scene
from paralleldomain.data_lab.config.reactor import ReactorConfig
from paralleldomain.data_lab.config.sensor_rig import SensorConfig, SensorRig
from paralleldomain.data_lab.reactor import (
    ReactorFrameStreamGenerator,
    ReactorInputLoader,
    change_shape,
    clear_output_path,
    get_instance_mask_and_prompts,
    get_mask_annotations,
    merge_instance_masks,
)
from paralleldomain.data_lab.sim_state import ExtendedSimState
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.decoder import DataStreamSceneDecoder
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.step.frame_decoder import StepFrameDecoder
from paralleldomain.decoding.step.scene_decoder import StepSceneDecoder
from paralleldomain.encoding.helper import get_encoding_format
from paralleldomain.encoding.pipeline_encoder import EncodingFormat
from paralleldomain.encoding.stream_pipeline_builder import StreamDatasetPipelineEncoder, StreamEncodingPipelineBuilder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType
from paralleldomain.model.frame import Frame
from paralleldomain.model.statistics.base import StatisticAliases, resolve_statistics
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.visualization import (
    get_active_recording_and_application_ids,
    set_active_recording_and_application_ids,
)
from paralleldomain.visualization.model_visualization import show_frame
from paralleldomain.visualization.state_visualization import show_agents, show_map
from paralleldomain.visualization.statistics.viewer import BACKEND

TimeOfDays = _env.TimeOfDays
Location = _loc.Location
CustomSimulationAgentBehavior = CustomSimulationAgentBehavior[ExtendedSimState]
CustomSimulationAgents = CustomSimulationAgents[ExtendedSimState]
CustomSimulationAgent = CustomSimulationAgent[ExtendedSimState]
CustomAtomicGenerator = CustomAtomicGenerator[ExtendedSimState]
encode_sim_states = encode_sim_states
_ = Scenario

# Maintained for backwards compatibility
CustomSimulationAgentBehaviour = CustomSimulationAgentBehavior

DEFAULT_DATA_LAB_VERSION = "v2.7.0+20231018CL65575"
"""Default Data Lab version to use across examples"""

coordinate_system = CoordinateSystem("RFU")
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
        self._last_recoding_id = None
        self._last_application_id = None

    def __call__(self, sim_state: ExtendedSimState):
        recording_id, application_id = get_active_recording_and_application_ids()

        if recording_id != self._last_recoding_id or application_id != self._last_application_id:
            show_map(umd_map=sim_state.map, recording_id=recording_id, application_id=application_id)
            self._last_recoding_id = recording_id
            self._last_application_id = application_id

        show_agents(sim_state=sim_state.current_state, recording_id=recording_id, application_id=application_id)

    def clone(self) -> "StateCallback":
        return SimStateVisualizerCallback(dataset_name=self.dataset_name)


def _create_decoded_stream_from_scenario(
    scenario_creator: ScenarioCreator,
    scene_index: int,
    simulator: Union[bool, SimulationStateProvider, str, None] = None,
    renderer: Union[bool, RenderInstance, str, None] = None,
    label_engine: Union[bool, LabelEngineInstance, str, None] = None,
    instance_name: Optional[str] = None,
    available_annotation_identifiers: Optional[
        List[AnnotationIdentifier]
    ] = None,  # required if label_engine_instance in kwargs
    auto_start_instance: bool = False,
    dataset_name: str = "Default Dataset Name",
    end_skip_frames: Optional[int] = None,
    frames_per_scene: Optional[int] = None,
    sim_capture_rate: Optional[int] = None,
    sim_settle_frames: Optional[int] = 40,
    start_skip_frames: Optional[int] = None,
    merge_batches: Optional[bool] = None,
    settings: Optional[DecoderSettings] = None,
    **kwargs,
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    if (renderer is None and instance_name is None and auto_start_instance is False) or renderer is False:
        raise ValueError(
            "Either a renderer or an instance name has to be passed in order to create frames with sensor data!"
        )

    gen = create_sensor_sim_stream(
        scenario_creator=scenario_creator,
        sim_settle_frames=sim_settle_frames,
        scene_index=scene_index,
        dataset_name=dataset_name,
        frames_per_scene=frames_per_scene,
        end_skip_frames=end_skip_frames,
        start_skip_frames=start_skip_frames,
        merge_batches=merge_batches,
        sim_capture_rate=sim_capture_rate,
        sim_state_type=ExtendedSimState,
        instance_name=instance_name,
        auto_start_instance=auto_start_instance,
        simulator=simulator,
        renderer=renderer,
        label_engine=label_engine,
        **kwargs,
    )
    scene_name = scene_index_to_name(scene_index=scene_index)
    scene = None
    og_scene = None
    decoder = None
    if settings is None:
        settings = DecoderSettings()
    for reference in gen:
        if scene is None:
            if isinstance(reference, LabeledStateReference):
                # New code path using label engine
                decoder = DataStreamSceneDecoder(
                    dataset_name=dataset_name,
                    settings=settings,
                    state_reference=reference,
                    available_annotation_identifiers=available_annotation_identifiers,
                    scene_name=scene_name,
                )
            else:
                # Old code path when label engine is not available
                decoder = StepSceneDecoder(
                    sensor_rig=reference.sensor_rig, dataset_name=dataset_name, settings=settings, scene_name=scene_name
                )

            scene = Scene(
                decoder=decoder,
            )
            og_scene = scene
            # TODO: previous code made pickling possible. Is that needed?

        if isinstance(reference, LabeledStateReference):
            assert isinstance(decoder, DataStreamSceneDecoder)
            decoder.update_labeled_state_reference(labeled_state_reference=reference)
            yield scene.get_frame(frame_id=reference.frame_id), scene
        else:
            assert isinstance(decoder, StepSceneDecoder)
            current_frame = Frame[datetime](
                decoder=StepFrameDecoder(
                    dataset_name=dataset_name,
                    date_time=reference.date_time,
                    ego_agent_id=reference.ego_agent_id,
                    frame_id=reference.frame_id,
                    scene_name=scene_name,
                    sensor_rig=reference.sensor_rig,
                    session=reference,
                    settings=DecoderSettings(),
                ),
            )
            decoder.add_frame(frame_id=reference.frame_id, date_time=reference.date_time)
            # cache all data in memory so that the sim and renderer can update
            in_memory_frame_decoder = InMemoryFrameDecoder.from_frame(frame=current_frame)
            yield Frame[datetime](decoder=in_memory_frame_decoder), scene
    if og_scene is not None:
        og_scene.clear_from_cache()


def create_frame_stream(
    scenario_creator: ScenarioCreator,
    dataset_name: str = "Default Dataset Name",
    scene_indices: List[int] = None,
    frames_per_scene: Optional[int] = None,
    statistic: StatisticAliases = None,
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    statistics_save_location: Union[str, AnyPath] = None,
    **kwargs,
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    if scene_indices is None:
        scene_indices = [0]
    if "number_of_scenes" in kwargs and scene_indices is None:
        number_of_scenes = kwargs.get("number_of_scenes")
        if number_of_scenes < 1:
            raise ValueError("A number of scenes > 0 has to be passed!")
        scene_indices = list(range(number_of_scenes))

    for scene_index in scene_indices:
        set_active_recording_and_application_ids(
            recording_id=f"{dataset_name}-{scene_index_to_name(scene_index=scene_index)}", application_id=dataset_name
        )
        for frame, scene in _create_decoded_stream_from_scenario(
            scenario_creator=scenario_creator,
            scene_index=scene_index,
            frames_per_scene=frames_per_scene,
            data_lab_version=data_lab_version,
            dataset_name=dataset_name,
            **kwargs,
        ):
            if statistic is not None:
                for sensor_frame in frame.sensor_frames:
                    statistic.update(scene=scene, sensor_frame=sensor_frame)
            yield frame, scene

    if statistic is not None and statistics_save_location is not None:
        statistics_save_location = AnyPath(statistics_save_location)
        with TemporaryDirectory() as f:
            statistic.save(path=f)
            AnyPath(f).copytree(statistics_save_location)


def create_reactor_frame_stream(
    scenario_creator: ScenarioCreator,
    reactor_config: ReactorConfig,
    scene_indices: List[int],
    use_cached_reactor_states: bool = False,
    instance_run_env: str = "sync",
    start_skip_frames: int = 5,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    output_path: Union[str, AnyPath, None] = None,
    available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
    **kwargs,
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    if output_path is not None:
        dataset_output_path = output_path
        logger.info(f"Output path is {dataset_output_path}")
    else:
        dataset_output_path = tempfile.mkdtemp()

    if instance_run_env != "sync":
        logger.warning(
            "Reactor will overwrite instance_run_env parameter to sync. "
            "Pypeln thread and process are currently not supported."
        )
    run_env = pypeln.sync

    if reactor_config.reactor_object.change_shape is True:
        # We want to visualize during the reactor run and not when caching the sim states,
        # so we need to transfer the SimStateVisualizerCallback to that scenario.
        sim_state_visualizer_callbacks = scenario_creator.find_state_callbacks(callback_type=SimStateVisualizerCallback)
        scenario_creator.remove_state_callbacks(state_callbacks=sim_state_visualizer_callbacks)
        simstate_dir = AnyPath(dataset_output_path + "/cache/sim_states")
        encode_dir = AnyPath(dataset_output_path + "/cache/dataset")
        if use_cached_reactor_states is True and simstate_dir.exists():
            if simstate_dir.is_cloud_path:
                raise NotImplementedError(
                    "Parameter use_cached_reactor_states is not supported for cloud datasets. Should be False"
                )
        else:
            # Remove renderer and label engine to encode sim states
            copy_kwargs = kwargs.copy()
            if "renderer" in copy_kwargs:
                copy_kwargs.pop("renderer", None)
            if "label_engine" in copy_kwargs:
                copy_kwargs.pop("label_engine", None)
            encode_sim_states(
                scenario_creator=scenario_creator,
                start_skip_frames=start_skip_frames,
                output_folder=simstate_dir,
                scene_indices=scene_indices,
                sim_state_type=ExtendedSimState,
                fail_on_sim_error=False,
                **copy_kwargs,
            )
        collection = SimulatedScenarioCollection(storage_folder=simstate_dir)

        # Exchange simulator with FromDiskSimulation, because we use stored sim states
        copy_kwargs = kwargs.copy()
        if "simulator" in copy_kwargs:
            copy_kwargs.pop("simulator", None)
        decoder_settings = None
        if reactor_config.distortion_lookups is not None and reactor_config.undistort_input is True:
            decoder_settings = DecoderSettings(distortion_lookups=reactor_config.distortion_lookups)

        # resolve label engine config name
        label_engine = copy_kwargs.get("label_engine", None)
        if label_engine is not None:
            label_engine_config_name = label_engine.config_name
        else:
            label_engine_config_name = DEFAULT_LABEL_ENGINE_CONFIG_NAME

        if use_cached_reactor_states is True and encode_dir.exists():
            stored_dataset = decode_dataset(
                dataset_path=encode_dir,
                dataset_format="data-stream",
                settings=decoder_settings,
                label_engine_config_name=label_engine_config_name,
            )
        else:
            logger.info("Reactor: creating auxiliary dataset.")
            simulated_scene_indices = list(set(scene_indices).intersection(set(collection.scene_indices)))
            _create_mini_batch(
                scenario_creator=collection,
                start_skip_frames=start_skip_frames,
                scene_indices=simulated_scene_indices,
                format="data-stream",
                output_path=encode_dir,
                simulator=FromDiskSimulation(),
                reactor_config=None,
                statistic=statistic,
                statistics_save_location=statistics_save_location,
                available_annotation_identifiers=available_annotation_identifiers,
                **copy_kwargs,
            )
            stored_dataset = decode_dataset(
                dataset_path=encode_dir,
                dataset_format="data-stream",
                settings=decoder_settings,
                label_engine_config_name=label_engine_config_name,
            )

        clear_output_path(output_path=AnyPath(dataset_output_path), scene_indices=scene_indices)
        filtered_collection = SimulatedScenarioCollection(storage_folder=simstate_dir)
        filtered_collection.add_state_callback(
            state_callback=FilterAsset(asset_name=reactor_config.reactor_object.asset_name)
        )

        reactor_input_loader = ReactorInputLoader(reactor_config=reactor_config, stored_dataset=stored_dataset)
        reactor_frame_stream_generator = ReactorFrameStreamGenerator(reactor_config=reactor_config)

        logger.info("Reactor: creating reactor dataset.")
        simulated_scene_indices = list(set(scene_indices).intersection(set(filtered_collection.scene_indices)))

        for state_callback in sim_state_visualizer_callbacks:
            filtered_collection.add_state_callback(state_callback=state_callback)
        yield from (
            create_frame_stream(
                scenario_creator=filtered_collection,
                start_skip_frames=start_skip_frames,
                scene_indices=simulated_scene_indices,
                simulator=FromDiskSimulation(),
                statistic=statistic,
                statistics_save_location=statistics_save_location,
                available_annotation_identifiers=available_annotation_identifiers,
                **copy_kwargs,
            )
            | run_env.map(f=reactor_input_loader.load_reactor_input, workers=1, maxsize=1)
            | run_env.map(f=reactor_frame_stream_generator.create_reactor_frame, workers=1, maxsize=1)
        )
    else:
        reactor_input_loader = ReactorInputLoader(reactor_config=reactor_config, stored_dataset=None)
        reactor_frame_stream_generator = ReactorFrameStreamGenerator(reactor_config=reactor_config)

        logger.info("Reactor: creating reactor dataset.")
        yield from (
            create_frame_stream(
                scenario_creator=scenario_creator,
                start_skip_frames=start_skip_frames,
                scene_indices=scene_indices,
                statistic=statistic,
                statistics_save_location=statistics_save_location,
                available_annotation_identifiers=available_annotation_identifiers,
                **kwargs,
            )
            | run_env.map(f=reactor_input_loader.load_reactor_input_rgbd, workers=1, maxsize=1)
            | run_env.map(f=reactor_frame_stream_generator.create_reactor_frame, workers=1, maxsize=1)
        )


def _get_local_encoder(
    scenario_creator: ScenarioCreator,
    scene_indices: List[int],
    frames_per_scene: int,
    output_path: Union[str, AnyPath, None] = None,
    instance_run_env: str = "thread",
    encoder_run_env: str = "thread",
    format: Union[str, EncodingFormat] = "dgpv1",
    use_tqdm: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    **kwargs,
) -> StreamDatasetPipelineEncoder:
    pipeline_kwargs = pipeline_kwargs if pipeline_kwargs is not None else dict()

    if frames_per_scene < 1:
        raise ValueError("A number of frames per scene > 0 has to be passed!")

    if isinstance(format, str):
        encoding_format = get_encoding_format(format_name=format, output_path=output_path)
    else:
        encoding_format = format
    if reactor_config is not None:
        frame_stream = create_reactor_frame_stream(
            scenario_creator=scenario_creator,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            instance_run_env=instance_run_env,
            output_path=output_path,
            **kwargs,
        )
    else:
        frame_stream = create_frame_stream(
            scenario_creator=scenario_creator,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            statistic=statistic,
            **kwargs,
        )
    pipeline_builder = StreamEncodingPipelineBuilder(
        frame_stream=frame_stream,
        run_env=encoder_run_env,
        number_of_frames_per_scene=frames_per_scene,
        **pipeline_kwargs,
    )
    encoder = StreamDatasetPipelineEncoder.from_builder(
        use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format
    )
    return encoder


def _create_mini_batch(
    scenario_creator: ScenarioCreator,
    scene_indices: List[int],
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    output_path: Union[str, AnyPath, None] = None,
    frames_per_scene: int = -1,
    instance_run_env: str = "thread",
    format: Union[str, EncodingFormat] = "dgpv1",
    instance_name: str = None,
    use_tqdm: bool = True,
    run_local: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    **kwargs,
):
    if run_local is True:
        encoder = _get_local_encoder(
            scenario_creator=scenario_creator,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            output_path=output_path,
            pipeline_kwargs=pipeline_kwargs,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            instance_name=instance_name,
            data_lab_version=data_lab_version,
            **kwargs,
        )
    else:
        encoder = _RemoteEncoder(
            provider=scenario_creator,
            name=instance_name,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            output_path=output_path,
            pipeline_kwargs=pipeline_kwargs,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            data_lab_version=data_lab_version,
            **kwargs,
        )
    encoder.encode_dataset()


class _RemoteEncoder:
    IS_INITIALIZED: bool = False
    _INIT_LOCK: Lock = Lock()

    def __init__(
        self,
        name: str,
        pem_file_content: str = None,
        simulation_only: bool = False,
        env_vars: Dict[str, str] = None,
        pip_requirements: Optional[List[str]] = None,
        ray_cluster_location: Optional[str] = None,
        **kwargs,
    ):
        ray_cluster_location = os.environ.get("PD_RAY_CLUSTER", ray_cluster_location)
        if pip_requirements is None:
            raise ValueError("Pip requirements not set. Can not install pd-sdk in remote cluster.")

        with _RemoteEncoder._INIT_LOCK:
            if not _RemoteEncoder.IS_INITIALIZED:
                runtime_env = dict()
                import ray

                with NamedTemporaryFile(suffix=".txt") as tmp_file:
                    if pip_requirements is not None:
                        tmp_file.writelines([s.encode("utf-8") for s in pip_requirements])
                    tmp_file.flush()
                    runtime_env["pip"] = tmp_file.name
                    runtime_env["eager_install"] = False
                    ray.init(address=ray_cluster_location, runtime_env=runtime_env)
                    _RemoteEncoder.IS_INITIALIZED = True

        pd.management.api_key = os.environ["PD_CLIENT_STEP_API_KEY_ENV"]
        pd.management.org = os.environ["PD_CLIENT_ORG_ENV"]
        try:
            data_lab_version = next(ig.ig_version for ig in Ig.list() if ig.name == name)
        except StopIteration:
            raise PdError(f"Your instance '{name}' could not be found. Please check your instance name and status.")

        self.kwargs = kwargs
        self.simulator_name = name
        self._simulation_only = simulation_only
        self.renderer_name = name if not simulation_only else None
        self.data_lab_version = data_lab_version
        if pem_file_content is None:
            pem_file_content = open(os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"]).read()
        self.pem_file_content = pem_file_content
        if env_vars is None:
            vars = ["PD_CLIENT_STEP_API_KEY_ENV", "PD_CLIENT_ORG_ENV"]
            env_vars = {name: os.environ[name] for name in vars}
        self.env_vars = env_vars

    def encode_dataset(self):
        import ray

        @ray.remote(num_cpus=2, max_retries=0)
        def _remote_create_mini_batch(env_vars: Dict[str, str], pem_file_content: str, **kwargs):
            os.environ.update(env_vars)
            with NamedTemporaryFile(suffix=".pem") as pem_file:
                with open(pem_file.name, "w") as file:
                    file.write(pem_file_content)

                os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"] = pem_file.name
                return _create_mini_batch(**kwargs)

        try:
            result = ray.get(
                _remote_create_mini_batch.remote(
                    simulator_name=self.simulator_name,
                    renderer_name=self.renderer_name,
                    env_vars=self.env_vars,
                    data_lab_version=self.data_lab_version,
                    pem_file_content=self.pem_file_content,
                    run_local=True,
                    **self.kwargs,
                )
            )
            return result
        except ray.exceptions.RayTaskError as e:
            logger.error(e)


def _get_instance_wise_kwargs(
    run_local: bool,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    simulator: Union[bool, SimulationStateProvider, str, None] = None,
    renderer: Union[bool, RenderInstance, str, None] = None,
    label_engine: Union[bool, LabelEngineInstance, str, None] = None,
    data_lab_instances: List[str] = None,
) -> List[Dict[str, Any]]:
    instance_wise_kwargs = []
    if not run_local:
        if data_lab_instances is None or len(data_lab_instances) == 0:
            instance_names = [
                i.name
                for i in [label_engine, simulator, renderer]
                if i is not None and not isinstance(i, bool) and i.name is not None
            ]
            if not len(set(instance_names)) == 1:
                raise ValueError(
                    "If no data_lab_instances names are passed and you want to encode remotely, "
                    "you cant use local instances!"
                )
            else:
                data_lab_instances = [instance_names[0]]
        for instance_name in data_lab_instances:
            instance_wise_kwargs.append(
                dict(
                    instance_name=instance_name,
                    pip_requirements=pip_requirements,
                    ray_cluster_location=ray_cluster_location,
                )
            )
    else:
        if data_lab_instances is None or len(data_lab_instances) == 0:
            instance_wise_kwargs.append(
                dict(
                    label_engine=label_engine,
                    simulator=simulator,
                    renderer=renderer,
                    instance_name=None,
                )
            )
        else:
            for instance_name in data_lab_instances:
                instance_wise_kwargs.append(
                    dict(
                        instance_name=instance_name,
                        label_engine=label_engine,
                        simulator=simulator,
                        renderer=renderer,
                    )
                )
    return instance_wise_kwargs


@overload
def create_mini_batch(
    scenario_creator: ScenarioCreator,
    output_path: Union[str, AnyPath, None] = None,
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    simulator: Union[bool, SimulationStateProvider, str, None] = None,
    renderer: Union[bool, RenderInstance, str, None] = None,
    label_engine: Union[bool, LabelEngineInstance, str, None] = None,
    instance_name: str = None,
    auto_start_instance: bool = False,
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    instance_run_env: str = "thread",
    format: Union[str, EncodingFormat] = "dgpv1",
    run_local: bool = True,
    use_tqdm: bool = True,
    debug: bool = False,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    scene_index_offset: Optional[int] = None,
    **kwargs,
):
    ...


@overload
def create_mini_batch(
    scenario_creator: ScenarioCreator,
    output_path: Union[str, AnyPath, None] = None,
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    simulator: bool = True,
    renderer: bool = True,
    label_engine: bool = True,
    data_lab_instances: Union[int, List[str]] = None,
    auto_start_instance: bool = False,
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    instance_run_env: str = "thread",
    format: Union[str, EncodingFormat] = "dgpv1",
    run_local: bool = True,
    use_tqdm: bool = True,
    debug: bool = False,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    scene_index_offset: Optional[int] = None,
    **kwargs,
):
    ...


def create_mini_batch(
    scenario_creator: ScenarioCreator,
    output_path: Union[str, AnyPath, None] = None,
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    simulator: Union[bool, SimulationStateProvider, str, None] = None,
    renderer: Union[bool, RenderInstance, str, None] = None,
    label_engine: Union[bool, LabelEngineInstance, str, None] = None,
    data_lab_instances: Union[int, List[str]] = None,
    instance_name: str = None,
    auto_start_instance: bool = False,
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    instance_run_env: str = "thread",
    format: Union[str, EncodingFormat] = "dgpv1",
    run_local: bool = True,
    use_tqdm: bool = True,
    debug: bool = False,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    scene_index_offset: Optional[int] = None,
    **kwargs,
):
    """

    Args:
        scenario_creator: The object used to create a scenario for a specific index.
        output_path: The output path for the dataset. Is only used if you pass a format at a string.
            You can also set this to None and pass a format object that defines its own output path.
            See DataStreamEncodingFormat.__init__
        number_of_scenes: The total number of scenes that should be generated.
        frames_per_scene: THe number of frames each scene should have.
            Note that each frame can contain data from several sensors
        simulator: You can provide either the name of the simulator instance to use, A SimulationProvider object
            in case you want to create or load custom simulation states, a boolean if you dont want to use a simulator
            or None (default) which will create a simulator from the passed data lab instance_name
        renderer: You can provide either the name of the renderer instance to use, A custom Renderer object,
            a boolean if you don't want to use a renderer (useful for simulation debugging since its faster)
            or None (default) which will create a renderer from the passed data lab instance_name
        label_engine: You can provide either the name of the label engine instance to use, A custom Label Engine object,
            a boolean if you don't want to use a label engine (not recommended)
            or None (default) which will create a label engine from the passed data lab instance_name
        data_lab_instances: A list of instances that you've already stared. If you pass this, you don't need to pass
            renderer, simulator or label_engine, only if you want to turn them off by setting them to false.
        instance_name: The name of the data lab instance to use. In case you are only using a single instance.
            If you pass this, you don't need to pass renderer, simulator or label_engine, instance_name,
            only if you want to turn them off by setting them to false.
        auto_start_instance: If you don't pass a data_lab_instance or instance_name etc, you can set this to true
            to automatically start a new instance.
        data_lab_version: The version of the data lab instance to use. Defaults to the latest version.
            Has to match your running instance.
        ray_cluster_location:
        pip_requirements:
        instance_run_env:
        format:
        run_local:
        use_tqdm:
        debug:
        reactor_config:
        use_cached_reactor_states:
        pipeline_kwargs:
        statistic:
        statistics_save_location:
        scene_index_offset:
        **kwargs:
    """
    if data_lab_instances is None and instance_name is not None:
        data_lab_instances = [instance_name]
    if isinstance(data_lab_instances, int) and auto_start_instance is True:
        data_lab_instances = [None for _ in range(data_lab_instances)]

    if (data_lab_instances is None or len(data_lab_instances) == 0) and renderer is None and simulator is None:
        raise ValueError(
            "You need to either pass a list of instance names via data_lab_instances or a"
            "renderer and simulator. In order to encode a dataset!"
        )

    if number_of_scenes < 1:
        raise ValueError("A number of scenes > 0 has to be passed!")
    scene_indices = list(range(number_of_scenes))

    if scene_index_offset is not None:
        scene_indices = [scene_idx + scene_index_offset for scene_idx in scene_indices]
    instance_wise_kwargs = _get_instance_wise_kwargs(
        run_local=run_local,
        pip_requirements=pip_requirements,
        ray_cluster_location=ray_cluster_location,
        label_engine=label_engine,
        simulator=simulator,
        renderer=renderer,
        data_lab_instances=data_lab_instances,
    )

    number_of_instances = len(instance_wise_kwargs)
    scene_index_split = [(scene_indices[i::number_of_instances], i) for i in range(number_of_instances)]

    runenv = pypeln.thread
    if debug:
        runenv = pypeln.sync

    generator = runenv.map(
        lambda scene_indices_split_and_i: _create_mini_batch(
            scenario_creator=scenario_creator,
            scene_indices=scene_indices_split_and_i[0],
            number_of_scenes=number_of_scenes,
            frames_per_scene=frames_per_scene,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            output_path=output_path,
            pipeline_kwargs=pipeline_kwargs,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            run_local=run_local,
            data_lab_version=data_lab_version,
            auto_start_instance=auto_start_instance,
            **instance_wise_kwargs[scene_indices_split_and_i[1]],
            **kwargs,
        ),
        scene_index_split,
        workers=number_of_instances,
    )
    runenv.run(generator)


def preview_scenario(
    scenario_creator: ScenarioCreator,
    dataset_name: str = "Default Dataset Name",
    simulator: Union[bool, SimulationStateProvider, str, None] = None,
    renderer: Union[bool, RenderInstance, str, None] = None,
    label_engine: Union[bool, LabelEngineInstance, str, None] = None,
    instance_name: Optional[str] = None,
    auto_start_instance: bool = False,
    data_lab_version: str = DEFAULT_DATA_LAB_VERSION,
    number_of_scenes: int = 1,
    frames_per_scene: int = 10,
    annotations_to_show: List[AnnotationType] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    **kwargs,
):
    if not any([isinstance(cb, SimStateVisualizerCallback) for cb in scenario_creator.state_callbacks]):
        scenario_creator.add_state_callback(SimStateVisualizerCallback())

    if statistic is not None:
        from paralleldomain.visualization.statistics.viewer import StatisticViewer

        statistic = resolve_statistics(statistics=statistic)
        _ = StatisticViewer.resolve_default_viewers(statistic=statistic, backend=BACKEND.RERUN)

    if renderer is None and instance_name is None and auto_start_instance is False:
        for scene_index in range(number_of_scenes):
            set_active_recording_and_application_ids(
                recording_id=f"{dataset_name}-{scene_index_to_name(scene_index=scene_index)}",
                application_id=dataset_name,
            )
            for _ in create_sensor_sim_stream(
                scenario_creator=scenario_creator,
                scene_index=scene_index,
                frames_per_scene=frames_per_scene,
                sim_state_type=ExtendedSimState,
                number_of_scenes=number_of_scenes,
                instance_name=instance_name,
                simulator=simulator,
                renderer=renderer,
                label_engine=label_engine,
                data_lab_version=data_lab_version,
                auto_start_instance=auto_start_instance,
                **kwargs,
            ):
                pass
    elif reactor_config is not None:
        for frame, scene in create_reactor_frame_stream(
            scenario_creator=scenario_creator,
            scene_indices=list(range(number_of_scenes)),
            frames_per_scene=frames_per_scene,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            number_of_scenes=number_of_scenes,
            instance_name=instance_name,
            simulator=simulator,
            renderer=renderer,
            label_engine=label_engine,
            data_lab_version=data_lab_version,
            auto_start_instance=auto_start_instance,
            dataset_name=dataset_name,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)
    else:
        for frame, _ in create_frame_stream(
            scenario_creator=scenario_creator,
            scene_indices=list(range(number_of_scenes)),
            frames_per_scene=frames_per_scene,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            number_of_scenes=number_of_scenes,
            instance_name=instance_name,
            simulator=simulator,
            renderer=renderer,
            label_engine=label_engine,
            data_lab_version=data_lab_version,
            auto_start_instance=auto_start_instance,
            dataset_name=dataset_name,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)


def save_sim_state_archive(
    scenario_creator: ScenarioCreator,
    scene_index: int,
    frames_per_scene: int,
    sim_capture_rate: int,
    simulator: SimulationInstance,
    output_path: AnyPath,
    yield_every_sim_state: bool = True,
    scenario_index_offset: int = 0,
    **kwargs,
) -> AnyPath:
    # Function returns AnyPath object of location where state file has been outputted

    gen = create_sensor_sim_stream(
        scenario_creator=scenario_creator,
        scene_index=scene_index,
        frames_per_scene=frames_per_scene,
        sim_capture_rate=sim_capture_rate,
        yield_every_sim_state=yield_every_sim_state,
        simulator=simulator,
        sim_state_type=ExtendedSimState,
        **kwargs,
    )

    archive_dir = AnyPath("state")
    scenario_file_path = AnyPath(f"scenario_{(scenario_index_offset + scene_index):09}.7z")

    if not output_path.is_cloud_path:
        output_path.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)

    for temporal_sensor_session_reference in gen:
        # serialize out our sim state
        state_bytes = state_to_bytes(state=temporal_sensor_session_reference.state)

        frame_id = temporal_sensor_session_reference.frame_id

        state_filepath = archive_dir / f"{frame_id}.pd"
        with state_filepath.open("wb") as f:
            f.write(state_bytes)

    with py7zr.SevenZipFile(str(scenario_file_path), "w") as zip_file:
        zip_file.writeall(path=str(archive_dir))

    for file in archive_dir.iterdir():
        file.rm(missing_ok=False)
    archive_dir.rmdir()

    scenario_file_path.copy(target=output_path / scenario_file_path)
    scenario_file_path.rm(missing_ok=False)

    logger.info(f"State file outputted to: {str(output_path / scenario_file_path)}")

    return output_path / scenario_file_path
