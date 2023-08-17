import logging
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union

from paralleldomain import Scene
from paralleldomain.decoding.in_memory.dataset_decoder import InMemoryDatasetDecoder
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.encoding.generic_pipeline_builder import GenericSceneAggregator
from paralleldomain.encoding.pipeline_encoder import (
    NAME_TO_RUNENV,
    DatasetPipelineEncoder,
    EncoderStep,
    EncodingFormat,
    PipelineBuilder,
)
from paralleldomain.encoding.stream_pipeline_item import StreamPipelineItem
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.frame import Frame
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import (
    CameraSensorFrame,
    FilePathedDataType,
    LidarSensorFrame,
    SensorDataCopyTypes,
    SensorFrame,
)
from paralleldomain.model.type_aliases import FrameId

logger = logging.getLogger(__name__)


class StreamGenericEncoderStep(EncoderStep):
    def __init__(
        self,
        encoding_format: EncodingFormat[StreamPipelineItem],
        fs_copy: bool,
        copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
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
                ] = sensor_frame.available_annotation_identifiers
                if isinstance(sensor_frame, CameraSensorFrame):
                    copy_data_types.append(FilePathedDataType.Image)
                if isinstance(sensor_frame, LidarSensorFrame):
                    copy_data_types.append(FilePathedDataType.PointCloud)
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
        pipeline_item: StreamPipelineItem,
    ) -> StreamPipelineItem:
        sensor_frame = pipeline_item.sensor_frame

        if sensor_frame is not None:
            for data_type in self._get_data_types_to_copy(sensor_frame=sensor_frame):
                data_or_path = None

                load_data = True
                if self.fs_copy:
                    if (
                        isinstance(data_type, AnnotationIdentifier)
                        and data_type in sensor_frame.available_annotation_identifiers
                    ):
                        file_path = sensor_frame.get_file_path(data_type=data_type)
                    elif issubclass(data_type, Image) and pipeline_item.camera_frame is not None:
                        file_path = sensor_frame.get_file_path(data_type=data_type)
                    elif issubclass(data_type, PointCloud) and pipeline_item.lidar_frame is not None:
                        file_path = sensor_frame.get_file_path(data_type=data_type)
                    else:
                        file_path = None

                    if file_path and self.encoding_format.supports_copy(
                        pipeline_item=pipeline_item, data_type=data_type, data_path=file_path
                    ):
                        data_or_path = file_path
                        load_data = False

                if load_data:
                    if (
                        isinstance(data_type, AnnotationIdentifier)
                        and data_type in sensor_frame.available_annotation_identifiers
                    ):
                        data_or_path = sensor_frame.get_annotations(annotation_identifier=data_type)
                    elif issubclass(data_type, Image) and pipeline_item.camera_frame is not None:
                        data_or_path = pipeline_item.camera_frame.image.rgba
                    elif issubclass(data_type, PointCloud) and pipeline_item.lidar_frame is not None:
                        data_or_path = pipeline_item.lidar_frame.point_cloud

                if data_or_path is not None:
                    self.encoding_format.save_data(pipeline_item=pipeline_item, data=data_or_path, data_type=data_type)

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


class StreamEncodingPipelineBuilder(PipelineBuilder[StreamPipelineItem]):
    def __init__(
        self,
        frame_stream: Generator[Tuple[Frame, Scene], None, None],
        number_of_scenes: int,
        number_of_frames_per_scene: int,
        scene_aggregator: EncoderStep = None,
        workers: int = 4,
        max_in_queue_size: int = 6,
        custom_encoder_steps: List[EncoderStep] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        allowed_frames: Optional[List[FrameId]] = None,
        copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
        should_copy_callbacks: Optional[
            Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
        ] = None,
        target_dataset_name: str = "MiniBatchDataset",
        copy_all_available_sensors_and_annotations: bool = False,
        run_env: Literal["thread", "process", "sync"] = "thread",
    ):
        self.number_of_scenes = number_of_scenes
        self.number_of_frames_per_scene = number_of_frames_per_scene
        self._seen_scene_frames = dict()
        self.workers = workers
        self.frame_stream = frame_stream
        self.max_in_queue_size = max_in_queue_size
        self.should_copy_callbacks = should_copy_callbacks
        self.target_dataset_name = target_dataset_name
        self.allowed_frames = allowed_frames
        self.sensor_names = sensor_names
        self.run_env = run_env
        self.copy_all_available_sensors_and_annotations = copy_all_available_sensors_and_annotations
        self.copy_data_types = copy_data_types

        self.scene_aggregator = scene_aggregator
        self.custom_encoder_steps = custom_encoder_steps

    def build_encoder_steps(self, encoding_format: EncodingFormat[StreamPipelineItem]) -> List[EncoderStep]:
        encoder_steps = list()
        if self.copy_data_types is not None or self.copy_all_available_sensors_and_annotations:
            encoder_steps.append(
                StreamGenericEncoderStep(
                    encoding_format=encoding_format,
                    copy_data_types=self.copy_data_types,
                    should_copy_callbacks=self.should_copy_callbacks,
                    fs_copy=False,
                    workers=self.workers,
                    in_queue_size=self.max_in_queue_size,
                    run_env=self.run_env,
                    copy_all_available_sensors_and_annotations=self.copy_all_available_sensors_and_annotations,
                )
            )
        if self.custom_encoder_steps is not None:
            for custom_step in self.custom_encoder_steps:
                encoder_steps.append(custom_step)

        if self.scene_aggregator is None:
            encoder_steps.append(GenericSceneAggregator(encoding_format=encoding_format, run_env=self.run_env))
        else:
            encoder_steps.append(self.scene_aggregator)
        return encoder_steps

    def _get_target_sensor_name(self, sensor_name: str):
        if self.sensor_names is None:
            return sensor_name
        elif isinstance(self.sensor_names, list):
            return sensor_name
        elif isinstance(self.sensor_names, dict):
            return self.sensor_names[sensor_name]
        else:
            raise ValueError(f"sensor_names is neither a list nor a dict but {type(self.sensor_names)}!")

    def build_pipeline_source_generator(self) -> Generator[StreamPipelineItem, None, None]:
        logger.info("Encoding Scene Step Stream")
        scene_reference_timestamp = datetime.min

        dataset_decoder = InMemoryDatasetDecoder(
            scene_names=[],
            unordered_scene_names=[],
            metadata=DatasetMeta(name=self.target_dataset_name, available_annotation_identifiers=list()),
        )
        scene_decoders = dict()
        scene_sensor_frames_count = dict()

        for frame, scene in self.frame_stream:
            scene_name = frame.scene_name
            if scene_name not in dataset_decoder.scene_names:
                dataset_decoder.scene_names.append(scene_name)
                dtypes = [
                    dt
                    for dt in scene.available_annotation_types
                    if dt not in dataset_decoder.metadata.available_annotation_types
                ]
                dataset_decoder.metadata.available_annotation_types.extend(dtypes)

            in_memory_frame_decoder = InMemoryFrameDecoder.from_frame(frame=frame)

            if scene_name not in self._seen_scene_frames:
                self._seen_scene_frames[scene_name] = 0

                scene_decoder = InMemorySceneDecoder.from_scene(scene)
                scene_decoders[scene_name] = scene_decoder
                scene_sensor_frames_count[scene_name] = 0

            self._seen_scene_frames[scene_name] += 1

            for sensor_frame in frame.sensor_frames:
                yield StreamPipelineItem(
                    frame_decoder=in_memory_frame_decoder,
                    sensor_name=sensor_frame.sensor_name,
                    frame_id=frame.frame_id,
                    scene_name=scene_name,
                    dataset_path=None,
                    dataset_format="step",
                    decoder_kwargs=dict(),
                    target_sensor_name=self._get_target_sensor_name(sensor_name=sensor_frame.sensor_name),
                    scene_reference_timestamp=scene_reference_timestamp,
                    dataset_decoder=dataset_decoder,
                    scene_decoder=scene_decoders[scene_name],
                    available_annotation_types=sensor_frame.available_annotation_types,
                )
                scene_sensor_frames_count[scene_name] += 1

            if self._seen_scene_frames[scene_name] >= self.number_of_frames_per_scene:
                # End of Scene
                yield StreamPipelineItem(
                    sensor_name=None,
                    frame_id=None,
                    scene_name=scene_name,
                    dataset_path=None,
                    dataset_format="step",
                    decoder_kwargs=dict(),
                    target_sensor_name=None,
                    scene_reference_timestamp=scene_reference_timestamp,
                    is_end_of_scene=True,
                    total_frames_in_scene=scene_sensor_frames_count[scene_name],
                    frame_decoder=None,
                    dataset_decoder=dataset_decoder,
                    scene_decoder=scene_decoders[scene_name],
                    available_annotation_types=scene.available_annotation_types,
                )

        yield StreamPipelineItem(
            sensor_name=None,
            frame_id=None,
            scene_name=None,
            dataset_path=None,
            dataset_format="step",
            decoder_kwargs=dict(),
            target_sensor_name=None,
            scene_reference_timestamp=None,
            is_end_of_dataset=True,
            total_scenes_in_dataset=len(scene_decoders),
            frame_decoder=None,
            dataset_decoder=dataset_decoder,
            scene_decoder=None,
            available_annotation_types=list(),
        )

    @property
    def pipeline_item_unit_name(self):
        return "sensor frames"


class StreamDatasetPipelineEncoder(DatasetPipelineEncoder):
    def yielding_encode_dataset(self) -> Generator[StreamPipelineItem, None, None]:
        stage = self.pipeline_builder.build_pipeline_source_generator()
        encoder_steps = self.pipeline_builder.build_encoder_steps(encoding_format=self.encoding_format)
        stage = self.build_pipeline(source_generator=stage, encoder_steps=encoder_steps)

        stage: Iterable[StreamPipelineItem] = self.run_env.to_iterable(stage)
        for item in stage:
            if item.sensor_frame is not None:
                yield item
                # yield item.sensor_frame
