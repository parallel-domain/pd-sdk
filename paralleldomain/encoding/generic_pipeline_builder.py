import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Literal, Optional, Type, Union

from paralleldomain import Dataset
from paralleldomain.encoding.pipeline_encoder import (
    NAME_TO_RUNENV,
    EncoderStep,
    EncodingFormat,
    PipelineBuilder,
    TPipelineItem,
)
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import (
    CameraSensorFrame,
    FilePathedDataType,
    LidarSensorFrame,
    RadarSensorFrame,
    SensorDataCopyTypes,
    SensorFrame,
)
from paralleldomain.model.type_aliases import FrameId

logger = logging.getLogger(__name__)


class GenericSceneAggregator(Generic[TPipelineItem], EncoderStep):
    def __init__(
        self,
        encoding_format: EncodingFormat[TPipelineItem],
        run_env: Literal["thread", "process", "sync"] = "thread",
        in_queue_size: int = 8,
        workers: int = 2,
    ):
        self.encoding_format = encoding_format
        self.workers = workers

        self.total_scenes = -1
        self.total_seen_scenes = 0
        self.total_seen_frames = 0
        self.total_sensor_frames = -1
        self._in_queue_size = in_queue_size
        self.seen_sensors_per_frame: Dict[FrameId, int] = defaultdict(lambda: 0)
        self.total_sensors_per_frame: Dict[FrameId, int] = defaultdict(lambda: -1)
        self.seen_sensor_frames_per_scene: Dict[str, int] = defaultdict(lambda: 0)
        self.total_sensor_frames_per_scene: Dict[str, int] = defaultdict(lambda: -1)
        self.encode_frame_run_env = NAME_TO_RUNENV[run_env]  # maps to pypeln thread, process or sync
        if run_env == "process":
            run_env = "thread"  # this step does not support multiprocessing
        self.run_env = NAME_TO_RUNENV[run_env]  # maps to pypeln thread, process or sync

    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        stage = input_stage
        stage = self.run_env.flat_map(f=self.count_frames, stage=stage, workers=1, maxsize=self._in_queue_size)
        stage = self.encode_frame_run_env.flat_map(
            f=self.encode_frame, stage=stage, workers=self.workers, maxsize=self._in_queue_size
        )
        stage = self.run_env.flat_map(
            f=self.encode_scene_and_dataset, stage=stage, workers=1, maxsize=self._in_queue_size
        )
        return stage

    def count_frames(self, pipeline_item: TPipelineItem) -> TPipelineItem:
        unique_frame_id = f"{pipeline_item.dataset_path}-{pipeline_item.scene_name}-{pipeline_item.frame_id}-"
        if not pipeline_item.is_end_of_frame:
            self.seen_sensors_per_frame[unique_frame_id] += 1
        else:
            self.total_sensor_frames_per_scene[unique_frame_id] = pipeline_item.total_sensors_in_frame

        total_sensors = self.total_sensor_frames_per_scene[unique_frame_id]
        total_seen_sensors = self.seen_sensors_per_frame[unique_frame_id]
        if total_sensors != -1 and total_seen_sensors != -1 and total_seen_sensors >= total_sensors:
            pipeline_item.custom_data["save_frame"] = True

        yield pipeline_item

    def encode_frame(self, pipeline_item: TPipelineItem) -> TPipelineItem:
        if "save_frame" in pipeline_item.custom_data and pipeline_item.custom_data["save_frame"] is True:
            self.encoding_format.save_frame(pipeline_item=pipeline_item)
        yield pipeline_item

    def encode_scene_and_dataset(self, pipeline_item: TPipelineItem) -> TPipelineItem:
        if not pipeline_item.is_end_of_frame:
            if not pipeline_item.is_end_of_scene:
                self.seen_sensor_frames_per_scene[pipeline_item.scene_name] += 1
            else:
                self.total_sensor_frames_per_scene[pipeline_item.scene_name] = pipeline_item.total_frames_in_scene

            total_frames = self.total_sensor_frames_per_scene[pipeline_item.scene_name]
            total_seen_frames = self.seen_sensor_frames_per_scene[pipeline_item.scene_name]
            if total_frames != -1 and total_seen_frames != -1 and total_seen_frames >= total_frames:
                self.encoding_format.save_scene(pipeline_item=pipeline_item)

            if pipeline_item.is_end_of_dataset:
                self.total_scenes = pipeline_item.total_scenes_in_dataset
            elif not pipeline_item.is_end_of_scene:
                self.total_seen_frames += 1
            else:
                if self.total_sensor_frames == -1:
                    self.total_sensor_frames = 0
                self.total_sensor_frames += pipeline_item.total_frames_in_scene
                self.total_seen_scenes += 1

            seen_all_frames = (
                self.total_sensor_frames != -1
                and self.total_seen_frames != -1
                and self.total_seen_frames >= self.total_sensor_frames
            )
            seen_all_scenes = (
                self.total_scenes != -1 and self.total_seen_scenes != -1 and self.total_seen_scenes >= self.total_scenes
            )
            if seen_all_frames and seen_all_scenes:
                self.encoding_format.save_dataset(pipeline_item=pipeline_item)

        if (
            not pipeline_item.is_end_of_scene
            and not pipeline_item.is_end_of_dataset
            and not pipeline_item.is_end_of_frame
        ):
            yield pipeline_item


class GenericEncoderStep(Generic[TPipelineItem], EncoderStep):
    def __init__(
        self,
        encoding_format: EncodingFormat[TPipelineItem],
        copy_data_types: List[SensorDataCopyTypes],
        fs_copy: bool,
        should_copy_callbacks: Optional[
            Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
        ] = None,
        workers: int = 1,
        in_queue_size: int = 1,
        run_env: Literal["thread", "process", "sync"] = "thread",
        copy_all_available_sensors_and_annotations: bool = False,
    ):
        self.copy_data_types = copy_data_types
        self.copy_all_available_sensors_and_annotations = copy_all_available_sensors_and_annotations
        self.should_copy_callbacks = should_copy_callbacks
        self.encoding_format = encoding_format
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.fs_copy = fs_copy
        self.run_env = NAME_TO_RUNENV[run_env]  # maps to pypeln thread, process or sync

    def _get_data_types_to_copy(
        self, sensor_frame: SensorFrame
    ) -> List[Union[SensorDataCopyTypes, AnnotationIdentifier]]:
        if self.copy_data_types is None:
            if self.copy_all_available_sensors_and_annotations:
                copy_data_types: List[
                    Union[SensorDataCopyTypes, AnnotationIdentifier]
                ] = sensor_frame.available_annotation_identifiers.copy()
                if isinstance(sensor_frame, CameraSensorFrame):
                    copy_data_types.append(FilePathedDataType.Image)
                elif isinstance(sensor_frame, LidarSensorFrame):
                    copy_data_types.append(FilePathedDataType.PointCloud)
                elif isinstance(sensor_frame, RadarSensorFrame):
                    copy_data_types.append(FilePathedDataType.RadarPointCloud)
                    copy_data_types.append(FilePathedDataType.RadarFrameHeader)
                    copy_data_types.append(FilePathedDataType.RangeDopplerMap)
            else:
                raise ValueError("Will encode Nothing!")
        else:
            copy_data_types = self.copy_data_types
        return [
            c
            for c in copy_data_types
            if self.should_copy_callbacks is None
            or c not in self.should_copy_callbacks
            or self.should_copy_callbacks[c](c, sensor_frame)
        ]

    def encode_frame_data(
        self,
        pipeline_item: TPipelineItem,
    ) -> TPipelineItem:
        sensor_frame = pipeline_item.sensor_frame

        if sensor_frame is not None:
            for data_type in self._get_data_types_to_copy(sensor_frame=sensor_frame):
                data_or_path = None

                load_data = True
                if self.fs_copy:
                    file_path = None
                    if isinstance(data_type, AnnotationIdentifier):
                        if data_type in sensor_frame.available_annotation_identifiers:
                            file_path = sensor_frame.get_file_path(data_type=data_type)
                    elif issubclass(data_type, Image) and pipeline_item.camera_frame is not None:
                        file_path = sensor_frame.get_file_path(data_type=data_type)
                    elif issubclass(data_type, PointCloud) and pipeline_item.lidar_frame is not None:
                        file_path = sensor_frame.get_file_path(data_type=data_type)

                    if file_path and self.encoding_format.supports_copy(
                        pipeline_item=pipeline_item, data_type=data_type, data_path=file_path
                    ):
                        data_or_path = file_path
                        load_data = False

                if load_data:
                    if isinstance(data_type, AnnotationIdentifier):
                        if data_type in sensor_frame.available_annotation_identifiers:
                            data_or_path = sensor_frame.get_annotations(annotation_identifier=data_type)
                    elif issubclass(data_type, Image) and pipeline_item.camera_frame is not None:
                        data_or_path = pipeline_item.camera_frame.image.rgba
                    elif issubclass(data_type, PointCloud) and pipeline_item.lidar_frame is not None:
                        data_or_path = pipeline_item.lidar_frame.point_cloud

                if data_or_path is not None:
                    self.encoding_format.save_data(pipeline_item=pipeline_item, data=data_or_path, data_type=data_type)
            self.encoding_format.save_sensor_frame(pipeline_item=pipeline_item)
        return pipeline_item

    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        stage = self.run_env.map(
            f=self.encode_frame_data,
            stage=stage,
            workers=self.workers,
            maxsize=self.in_queue_size,
        )
        return stage


class GenericPipelineBuilder(PipelineBuilder[TPipelineItem]):
    def __init__(
        self,
        pipeline_item_type: Type[TPipelineItem],
        dataset: Dataset,
        scene_aggregator: EncoderStep = None,
        workers: int = 2,
        max_in_queue_size: int = 6,
        inplace: bool = False,
        use_default_encoder_step: bool = True,
        custom_encoder_steps: List[EncoderStep] = None,
        sensor_name_mapping: Optional[Dict[str, str]] = None,
        copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
        should_copy_callbacks: Optional[
            Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
        ] = None,
        fs_copy: bool = True,
        copy_all_available_sensors_and_annotations: bool = False,
        run_env: Literal["thread", "process", "sync"] = "thread",
    ):
        self.use_default_encoder_step = use_default_encoder_step
        self.pipeline_item_type = pipeline_item_type
        self.workers = workers
        self.max_in_queue_size = max_in_queue_size
        self.fs_copy = fs_copy
        self.inplace = inplace
        self.should_copy_callbacks = should_copy_callbacks
        self.sensor_name_mapping = sensor_name_mapping
        self.run_env = run_env

        self.copy_data_types = copy_data_types
        self.copy_all_available_sensors_and_annotations = copy_all_available_sensors_and_annotations

        self._dataset = dataset
        self._scene_names = self._dataset.unordered_scene_names

        self.scene_aggregator = scene_aggregator
        self.custom_encoder_steps = custom_encoder_steps

    def build_encoder_steps(self, encoding_format: EncodingFormat[TPipelineItem]) -> List[EncoderStep]:
        encoder_steps = list()
        if self.use_default_encoder_step is True:
            encoder_steps.append(
                GenericEncoderStep(
                    encoding_format=encoding_format,
                    copy_data_types=self.copy_data_types,
                    should_copy_callbacks=self.should_copy_callbacks,
                    copy_all_available_sensors_and_annotations=self.copy_all_available_sensors_and_annotations,
                    fs_copy=self.fs_copy,
                    workers=self.workers,
                    in_queue_size=self.max_in_queue_size,
                    run_env=self.run_env,
                )
            )
        if self.custom_encoder_steps is not None:
            for custom_step in self.custom_encoder_steps:
                encoder_steps.append(custom_step)

        if self.scene_aggregator is None:
            encoder_steps.append(
                GenericSceneAggregator(
                    encoding_format=encoding_format,
                    run_env=self.run_env,
                    in_queue_size=self.max_in_queue_size,
                    workers=self.workers,
                )
            )
        else:
            encoder_steps.append(self.scene_aggregator)
        return encoder_steps

    def build_pipeline_source_generator(self) -> Generator[TPipelineItem, None, None]:
        dataset = self._dataset
        for scene_name in self._scene_names:
            scene = self._dataset.get_unordered_scene(scene_name=scene_name)

            if self.sensor_name_mapping is None:
                sensor_name_mapping = {s: s for s in scene.sensor_names}
            else:
                sensor_name_mapping = {s: t for s, t in self.sensor_name_mapping.items() if s in scene.sensor_names}

            reference_timestamp: datetime = scene.get_frame(list(scene.frame_ids)[0]).date_time

            total_frames_in_scene = 0
            logger.info(f"Encoding Scene {scene.name} with sensor mapping: {sensor_name_mapping}")
            # for target_sensor_name, source_sensor_name in sensor_name_mapping.items():
            for frame_id in scene.frame_ids:
                frame = scene.get_frame(frame_id=frame_id)
                available_sensor_names = frame.sensor_names
                for sensor_name in available_sensor_names:
                    if sensor_name in sensor_name_mapping.values():
                        yield self.pipeline_item_type(
                            sensor_name=sensor_name,
                            frame_id=frame_id,
                            scene_name=scene.name,
                            dataset_path=dataset.path,
                            dataset_format=dataset.format,
                            decoder_kwargs=dataset.decoder_init_kwargs,
                            target_sensor_name=sensor_name_mapping[sensor_name],
                            scene_reference_timestamp=reference_timestamp,
                        )
                        total_frames_in_scene += 1
                yield self.pipeline_item_type(
                    sensor_name=None,
                    frame_id=frame_id,
                    scene_name=scene.name,
                    dataset_path=dataset.path,
                    dataset_format=dataset.format,
                    decoder_kwargs=dataset.decoder_init_kwargs,
                    target_sensor_name=None,
                    scene_reference_timestamp=reference_timestamp,
                    is_end_of_frame=True,
                    total_sensors_in_frame=len(available_sensor_names),
                )
            yield self.pipeline_item_type(
                sensor_name=None,
                frame_id=None,
                scene_name=scene.name,
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                target_sensor_name=None,
                scene_reference_timestamp=reference_timestamp,
                is_end_of_scene=True,
                total_frames_in_scene=total_frames_in_scene,
            )
        yield self.pipeline_item_type(
            sensor_name=None,
            frame_id=None,
            scene_name=None,
            dataset_path=dataset.path,
            dataset_format=dataset.format,
            decoder_kwargs=dataset.decoder_init_kwargs,
            target_sensor_name=None,
            scene_reference_timestamp=None,
            is_end_of_dataset=True,
            total_scenes_in_dataset=len(self._scene_names),
        )

    @property
    def pipeline_item_unit_name(self):
        return "sensor frames"
