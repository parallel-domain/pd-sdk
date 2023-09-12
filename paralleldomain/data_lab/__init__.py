import logging
import os
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from tempfile import TemporaryDirectory, NamedTemporaryFile
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import py7zr
import pypeln

import pd.data_lab.config.environment as _env
import pd.data_lab.config.location as _loc
import pd.management
from paralleldomain import Scene
from paralleldomain.data_lab.config.reactor import ReactorConfig
from paralleldomain.data_lab.config.sensor_rig import SensorConfig, SensorRig
from paralleldomain.data_lab.reactor import (
    ReactorFrameStreamGenerator,
    ReactorInputLoader,
    clear_output_path,
    get_instance_mask_and_prompts,
    get_mask_annotations,
    change_shape,
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
from paralleldomain.encoding.stream_pipeline_builder import StreamDatasetPipelineEncoder, StreamEncodingPipelineBuilder
from paralleldomain.model.annotation import AnnotationType, AnnotationIdentifier
from paralleldomain.model.frame import Frame
from paralleldomain.model.statistics.base import resolve_statistics, StatisticAliases
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.visualization.model_visualization import show_frame
from paralleldomain.visualization.state_visualization import show_agents, show_map
from paralleldomain.visualization.statistics.viewer import BACKEND
from pd.data_lab import (
    TSimState,
    create_sensor_sim_stream,
    encode_sim_states,
    LabeledStateReference,
)
from pd.data_lab.context import setup_datalab, datalab_context_exists
from pd.data_lab.generators.custom_generator import CustomAtomicGenerator
from pd.data_lab.generators.custom_simulation_agent import (
    CustomSimulationAgent,
    CustomSimulationAgentBehaviour,
    CustomSimulationAgents,
)
from pd.data_lab.label_engine_instance import LabelEngineInstance
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Scenario, ScenarioSource, SimulatedScenarioCollection
from pd.data_lab.sim_instance import FromDiskSimulation
from pd.data_lab.sim_instance import SimulationInstance
from pd.data_lab.state_callback import StateCallback
from pd.management import Ig
from pd.state import ModelAgent, VehicleAgent
from pd.state import state_to_bytes

TimeOfDays = _env.TimeOfDays
Location = _loc.Location
CustomSimulationAgentBehaviour = CustomSimulationAgentBehaviour[ExtendedSimState]
CustomSimulationAgents = CustomSimulationAgents[ExtendedSimState]
CustomSimulationAgent = CustomSimulationAgent[ExtendedSimState]
CustomAtomicGenerator = CustomAtomicGenerator[ExtendedSimState]
encode_sim_states = encode_sim_states
_ = Scenario


DEFAULT_DATA_LAB_VERSION = "v2.5.0-beta"
"""Default Data Lab version to use across examples"""

# TODO: LFU to FLU (left to right hand problem)
# coordinate_system = INTERNAL_COORDINATE_SYSTEM
coordinate_system = CoordinateSystem("RFU")
logger = logging.getLogger(__name__)


def _get_instance_name(
    sim_instance_name: Optional[str] = None,
    le_instance_name: Optional[str] = None,
    render_instance_name: Optional[str] = None,
    instance_name: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    names = [sim_instance_name, le_instance_name, render_instance_name, instance_name]
    names = [n for n in names if n is not None]
    instances = ["render_instance", "sim_instance", "label_engine_instance"]
    names += [kwargs[k].name for k in instances if k in kwargs and kwargs[k] is not None and hasattr(kwargs[k], "name")]
    return next(iter(names), None)


@contextmanager
def _data_lab_context(
    env_vars: Dict[str, str], pem_file_content: Optional[str], data_lab_version: str, **kwargs
) -> Generator[None, None, None]:
    if data_lab_version is None:
        name = _get_instance_name(**kwargs)
        if name is not None:
            pd.management.api_key = os.environ["PD_CLIENT_STEP_API_KEY_ENV"]
            pd.management.org = os.environ["PD_CLIENT_ORG_ENV"]
            try:
                data_lab_version = next(ig.ig_version for ig in Ig.list() if ig.name == name)
            except StopIteration:
                raise ValueError(f"Unknown instance name {name}. Not found!")
        else:
            raise ValueError("Please pass a data_lab_version!")

    os.environ.update(env_vars)
    if datalab_context_exists() is True:
        yield
    elif pem_file_content is not None:
        with NamedTemporaryFile(suffix=".pem") as pem_file:
            with open(pem_file.name, "w") as file:
                file.write(pem_file_content)

            os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"] = pem_file.name
            setup_datalab(data_lab_version)
            yield
    else:
        setup_datalab(data_lab_version)
        yield


def set_data_lab_context(func):
    @wraps(func)
    def with_context(*args, **kwargs):
        setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
        with _data_lab_context(
            env_vars=kwargs.pop("env_vars", dict()),
            pem_file_content=kwargs.pop("pem_file_content", None),
            data_lab_version=kwargs.pop("data_lab_version", None),
            **kwargs,
        ):
            return func(*args, **kwargs)

    return with_context


def _resolve_name_to_instances(
    use_label_engine: bool = False,
    sim_instance_name: Optional[str] = None,
    le_instance_name: Optional[str] = None,
    render_instance_name: Optional[str] = None,
    instance_name: Optional[str] = None,
    **kwargs,
) -> Tuple[RenderInstance, SimulationInstance, LabelEngineInstance]:
    if (
        instance_name is not None
        and sim_instance_name is None
        and le_instance_name is None
        and render_instance_name is None
    ):
        render_instance_name = instance_name
        le_instance_name = instance_name
        sim_instance_name = instance_name

    render_instance = RenderInstance(name=render_instance_name) if render_instance_name is not None else None
    label_engine_instance = (
        LabelEngineInstance(name=le_instance_name) if use_label_engine and le_instance_name is not None else None
    )
    sim_instance = SimulationInstance(name=sim_instance_name) if sim_instance_name is not None else None
    return render_instance, sim_instance, label_engine_instance


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
    scene_index: int,
    available_annotation_identifiers: Optional[
        List[AnnotationIdentifier]
    ] = None,  # required if label_engine_instance in kwargs
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
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    render_instance, sim_instance, label_engine_instance = _resolve_name_to_instances(**kwargs)
    if render_instance is not None:
        kwargs["render_instance"] = render_instance
    if sim_instance is not None:
        kwargs["sim_instance"] = sim_instance
    if label_engine_instance is not None:
        kwargs["label_engine_instance"] = label_engine_instance

    discrete_scenario, gen = create_sensor_sim_stream(
        scenario=scenario,
        sim_settle_frames=sim_settle_frames,
        scene_index=scene_index,
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
    decoder = None
    for reference in gen:
        if scene is None:
            if isinstance(reference, LabeledStateReference):
                # New code path using label engine
                decoder = DataStreamSceneDecoder(
                    dataset_name=dataset_name,
                    settings=DecoderSettings(),
                    state_reference=reference,
                    available_annotation_identifiers=available_annotation_identifiers,
                )
            else:
                # Old code path when label engine is not available
                decoder = StepSceneDecoder(
                    sensor_rig=reference.sensor_rig,
                    dataset_name=dataset_name,
                    settings=DecoderSettings(),
                )

            scene = Scene(
                decoder=decoder,
                name=scene_name,
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
                frame_id=reference.frame_id,
                decoder=StepFrameDecoder(
                    dataset_name=dataset_name,
                    date_time=reference.date_time,
                    ego_agent_id=reference.ego_agent_id,
                    scene_name=scene_name,
                    sensor_rig=reference.sensor_rig,
                    session=reference,
                    settings=DecoderSettings(),
                ),
            )
            decoder.add_frame(frame_id=reference.frame_id, date_time=reference.date_time)
            # cache all data in memory so that the sim and renderer can update
            in_memory_frame_decoder = InMemoryFrameDecoder.from_frame(frame=current_frame)
            yield Frame[datetime](
                frame_id=current_frame.frame_id,
                decoder=in_memory_frame_decoder,
            ), scene
    if og_scene is not None:
        og_scene.clear_from_cache()


def create_frame_stream(
    scenario: ScenarioSource,
    scene_indices: List[int] = None,
    frames_per_scene: Optional[int] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    **kwargs,
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    if scene_indices is None:
        scene_indices = [0]
    if "number_of_scenes" in kwargs:
        logger.warning("Deprecated parameter number_of_scenes. Use scene_indices instead.")
        number_of_scenes = kwargs.pop("number_of_scenes")
        if number_of_scenes < 1:
            raise ValueError("A number of scenes > 0 has to be passed!")
        scene_indices = list(range(number_of_scenes))

    for scene_index in scene_indices:
        for frame, scene in _create_decoded_stream_from_scenario(
            scenario=scenario, scene_index=scene_index, frames_per_scene=frames_per_scene, **kwargs
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
    scenario: ScenarioSource,
    reactor_config: ReactorConfig,
    scene_indices: List[int],
    format_kwargs: Dict[str, Any],
    use_cached_reactor_states: bool = False,
    instance_run_env: str = "sync",
    sim_instance: SimulationInstance = None,
    render_instance: RenderInstance = None,
    sim_instance_name: str = None,
    render_instance_name: str = None,
    instance_name: str = None,
    **kwargs,
) -> Generator[Tuple[Frame[datetime], Scene], None, None]:
    dataset_output_path = format_kwargs["dataset_output_path"]
    logger.info(f"Output path is {dataset_output_path}")
    simstate_dir = AnyPath(dataset_output_path + "/cache/sim_states")
    encode_dir = AnyPath(dataset_output_path + "/cache/dataset")

    if render_instance is None and sim_instance is None:
        render_instance, sim_instance, _ = _resolve_name_to_instances(
            sim_instance_name=sim_instance_name,
            render_instance_name=render_instance_name,
            instance_name=instance_name,
            **kwargs,
        )

    if use_cached_reactor_states is True and simstate_dir.exists():
        if simstate_dir.is_cloud_path:
            raise NotImplementedError(
                "Parameter use_cached_reactor_states is not supported for cloud datasets. Should be False"
            )
        collection = SimulatedScenarioCollection(storage_folder=simstate_dir)
    else:
        encode_sim_states(
            scenario=scenario,
            output_folder=simstate_dir,
            scene_indices=scene_indices,
            sim_instance=sim_instance,
            render_instance=None,
            sim_state_type=ExtendedSimState,
            fail_on_sim_error=False,
            **kwargs,
        )
        collection = SimulatedScenarioCollection(storage_folder=simstate_dir)

    if use_cached_reactor_states is True and encode_dir.exists():
        stored_dataset = decode_dataset(dataset_path=encode_dir, dataset_format="dgpv1")
    else:
        logger.info("Reactor: creating auxiliary dataset.")
        simulated_scene_indices = list(set(scene_indices).intersection(set(collection.scene_indices)))
        _create_mini_batch(
            scenario=collection,
            scene_indices=simulated_scene_indices,
            format_kwargs=dict(
                dataset_output_path=encode_dir,
                encode_to_binary=False,
            ),
            sim_instance=FromDiskSimulation(),
            render_instance=render_instance,
            pipeline_kwargs=dict(copy_all_available_sensors_and_annotations=True, run_env="thread"),
            reactor_config=None,
            **kwargs,
        )
        stored_dataset = decode_dataset(dataset_path=encode_dir, dataset_format="dgpv1")

    clear_output_path(output_path=AnyPath(dataset_output_path), scene_indices=scene_indices)
    filtered_collection = SimulatedScenarioCollection(storage_folder=simstate_dir)
    filtered_collection.add_state_callback(
        state_callback=FilterAsset(asset_name=reactor_config.reactor_object.asset_name)
    )

    reactor_input_loader = ReactorInputLoader(reactor_config=reactor_config, stored_dataset=stored_dataset)
    reactor_frame_stream_generator = ReactorFrameStreamGenerator(reactor_config=reactor_config)

    logger.info("Reactor: creating reactor dataset.")
    if instance_run_env == "thread":
        run_env = pypeln.thread
    elif instance_run_env == "process":
        run_env = pypeln.process
    else:
        run_env = pypeln.sync

    simulated_scene_indices = list(set(scene_indices).intersection(set(filtered_collection.scene_indices)))

    yield from (
        create_frame_stream(
            scenario=filtered_collection,
            scene_indices=simulated_scene_indices,
            sim_instance=FromDiskSimulation(),
            render_instance=render_instance,
            instance_run_env=instance_run_env,
            **kwargs,
        )
        | run_env.map(f=reactor_input_loader.load_reactor_input, workers=1, maxsize=1)
        | run_env.map(f=reactor_frame_stream_generator.create_reactor_frame, workers=1, maxsize=1)
    )


def _get_local_encoder(
    scenario: ScenarioSource,
    scene_indices: List[int],
    frames_per_scene: int,
    instance_run_env: str = "thread",
    format: str = "dgpv1",
    use_tqdm: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    **kwargs,
) -> StreamDatasetPipelineEncoder:
    format_kwargs = format_kwargs if format_kwargs is not None else dict()
    pipeline_kwargs = pipeline_kwargs if pipeline_kwargs is not None else dict()

    if frames_per_scene < 1:
        raise ValueError("A number of frames per scene > 0 has to be passed!")

    encoding_format = get_encoding_format(format_name=format, **format_kwargs)
    if reactor_config is not None:
        frame_stream = create_reactor_frame_stream(
            scenario=scenario,
            scene_indices=scene_indices,
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
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            instance_run_env=instance_run_env,
            statistic=statistic,
            **kwargs,
        )
    pipeline_builder = StreamEncodingPipelineBuilder(
        frame_stream=frame_stream,
        number_of_frames_per_scene=frames_per_scene,
        **pipeline_kwargs,
    )
    encoder = StreamDatasetPipelineEncoder.from_builder(
        use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format
    )
    return encoder


@set_data_lab_context
def _create_mini_batch(
    scenario: ScenarioSource,
    scene_indices: List[int],
    frames_per_scene: int = -1,
    instance_run_env: str = "thread",
    scene_name_gen: Callable[[int], str] = None,
    format: str = "dgpv1",
    instance_name: str = None,
    use_tqdm: bool = True,
    run_local: bool = True,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    **kwargs,
):
    if run_local is True:
        encoder = _get_local_encoder(
            scenario=scenario,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            scene_name_gen=scene_name_gen,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            format_kwargs=format_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            instance_name=instance_name,
            **kwargs,
        )
    else:
        encoder = _RemoteEncoder(
            scenario=scenario,
            name=instance_name,
            scene_indices=scene_indices,
            frames_per_scene=frames_per_scene,
            scene_name_gen=scene_name_gen,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            format_kwargs=format_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
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
            raise ValueError(f"Unknown instance name {name}. Not found!")

        self.kwargs = kwargs
        self.sim_instance_name = name
        self._simulation_only = simulation_only
        self.render_instance_name = name if not simulation_only else None
        self.data_lab_version = data_lab_version
        if pem_file_content is None:
            pem_file_content = open(os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"], "r").read()
        self.pem_file_content = pem_file_content
        if env_vars is None:
            vars = ["PD_CLIENT_STEP_API_KEY_ENV", "PD_CLIENT_ORG_ENV"]
            env_vars = {name: os.environ[name] for name in vars}
        self.env_vars = env_vars

    def encode_dataset(self):
        import ray

        @ray.remote(num_cpus=1)
        def _remote_create_mini_batch(**kwargs):
            return _create_mini_batch(**kwargs)

        result = ray.get(
            _remote_create_mini_batch.remote(
                sim_instance_name=self.sim_instance_name,
                render_instance_name=self.render_instance_name,
                env_vars=self.env_vars,
                data_lab_version=self.data_lab_version,
                pem_file_content=self.pem_file_content,
                run_local=True,
                **self.kwargs,
            )
        )
        return result


def _get_instance_wise_kwargs(
    run_local: bool,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    label_engine_instance: Optional[LabelEngineInstance] = None,
    sim_instance: Optional[SimulationInstance] = None,
    render_instance: Optional[RenderInstance] = None,
    data_lab_instances: List[str] = None,
) -> List[Dict[str, Any]]:
    instance_wise_kwargs = []
    if not run_local:
        if data_lab_instances is None or len(data_lab_instances) == 0:
            instance_names = [
                i.name
                for i in [label_engine_instance, sim_instance, render_instance]
                if i is not None and i.name is not None
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
                    label_engine_instance=label_engine_instance,
                    sim_instance=sim_instance,
                    render_instance=render_instance,
                )
            )
        else:
            for instance_name in data_lab_instances:
                instance_wise_kwargs.append(
                    dict(
                        instance_name=instance_name,
                    )
                )
    return instance_wise_kwargs


def create_mini_batch(
    scenario: ScenarioSource,
    number_of_scenes: int = -1,
    frames_per_scene: int = -1,
    label_engine_instance: Optional[LabelEngineInstance] = None,
    sim_instance: Optional[SimulationInstance] = None,
    render_instance: Optional[RenderInstance] = None,
    data_lab_instances: List[str] = None,
    ray_cluster_location: str = None,
    pip_requirements: List[str] = None,
    instance_run_env: str = "thread",
    format: str = "dgpv1",
    run_local: bool = True,
    use_tqdm: bool = True,
    use_label_engine: bool = False,
    debug: bool = False,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    format_kwargs: Dict[str, Any] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    **kwargs,
):
    if (
        (data_lab_instances is None or len(data_lab_instances) == 0)
        and render_instance is None
        and sim_instance is None
    ):
        raise ValueError(
            "You need to either pass a list of instance names via data_lab_instances or a"
            "render_instance and sim_instance. In order to encode a dataset!"
        )

    if number_of_scenes < 1:
        raise ValueError("A number of scenes > 0 has to be passed!")
    scene_indices = list(range(number_of_scenes))

    instance_wise_kwargs = _get_instance_wise_kwargs(
        run_local=run_local,
        pip_requirements=pip_requirements,
        ray_cluster_location=ray_cluster_location,
        label_engine_instance=label_engine_instance,
        sim_instance=sim_instance,
        render_instance=render_instance,
        data_lab_instances=data_lab_instances,
    )

    number_of_instances = len(instance_wise_kwargs)
    scene_index_split = [(scene_indices[i::number_of_instances], i) for i in range(number_of_instances)]

    runenv = pypeln.thread
    if debug:
        runenv = pypeln.sync

    generator = runenv.map(
        lambda scene_indices_split_and_i: _create_mini_batch(
            scenario=scenario,
            scene_indices=scene_indices_split_and_i[0],
            frames_per_scene=frames_per_scene,
            format=format,
            instance_run_env=instance_run_env,
            use_tqdm=use_tqdm,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            format_kwargs=format_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            use_label_engine=use_label_engine,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            run_local=run_local,
            **instance_wise_kwargs[scene_indices_split_and_i[1]],
            **kwargs,
        ),
        scene_index_split,
        workers=number_of_instances,
    )
    runenv.run(generator)


def preview_scenario(
    scenario: ScenarioSource,
    number_of_scenes: int = 1,
    frames_per_scene: int = 10,
    annotations_to_show: List[AnnotationType] = None,
    statistic: StatisticAliases = None,
    statistics_save_location: Union[str, AnyPath] = None,
    reactor_config: ReactorConfig = None,
    use_cached_reactor_states: bool = False,
    use_label_engine: bool = False,
    **kwargs,
):
    if not any([isinstance(cb, SimStateVisualizerCallback) for cb in scenario.state_callbacks]):
        scenario.add_state_callback(SimStateVisualizerCallback())

    if statistic is not None:
        from paralleldomain.visualization.statistics.viewer import StatisticViewer

        statistic = resolve_statistics(statistics=statistic)
        _ = StatisticViewer.resolve_default_viewers(statistic=statistic, backend=BACKEND.RERUN)

    if kwargs.get("render_instance", None) is None:
        for scene_index in range(number_of_scenes):
            _, gen = create_sensor_sim_stream(
                scenario=scenario,
                scene_index=scene_index,
                frames_per_scene=frames_per_scene,
                sim_state_type=ExtendedSimState,
                **kwargs,
            )
            for _ in gen:
                pass
    elif reactor_config is not None:
        for frame, scene in create_reactor_frame_stream(
            scenario=scenario,
            scene_indices=list(range(number_of_scenes)),
            frames_per_scene=frames_per_scene,
            reactor_config=reactor_config,
            use_cached_reactor_states=use_cached_reactor_states,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            use_label_engine=use_label_engine,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)
    else:
        for frame, _ in create_frame_stream(
            scenario=scenario,
            scene_indices=list(range(number_of_scenes)),
            frames_per_scene=frames_per_scene,
            statistic=statistic,
            statistics_save_location=statistics_save_location,
            use_label_engine=use_label_engine,
            **kwargs,
        ):
            show_frame(frame=frame, annotations_to_show=annotations_to_show)


def save_sim_state_archive(
    scenario: Scenario,
    scene_index: int,
    frames_per_scene: int,
    sim_capture_rate: int,
    sim_instance: SimulationInstance,
    output_path: AnyPath,
    yield_every_sim_state: bool = True,
    scenario_index_offset: int = 0,
    **kwargs,
) -> AnyPath:
    # Function returns AnyPath object of location where state file has been outputted

    discrete_scenario, gen = create_sensor_sim_stream(
        scenario=scenario,
        scene_index=scene_index,
        frames_per_scene=frames_per_scene,
        sim_capture_rate=sim_capture_rate,
        yield_every_sim_state=yield_every_sim_state,
        sim_instance=sim_instance,
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
