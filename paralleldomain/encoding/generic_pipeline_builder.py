import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Optional, Type, Union

import pypeln

from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.encoding.pipeline_encoder import EncoderStep, EncodingFormat, PipelineBuilder, TPipelineItem
from paralleldomain.model.annotation import Annotation
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import FilePathedDataType, SensorDataTypes, SensorFrame
from paralleldomain.model.type_aliases import FrameId
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class GenericSceneAggregator(Generic[TPipelineItem], EncoderStep):
    def __init__(self, encoding_format: EncodingFormat[TPipelineItem]):
        self.encoding_format = encoding_format

        self.total_scenes = -1
        self.total_seen_scenes = 0
        self.total_seen_frames = 0
        self.total_frames = -1
        self.seen_frames_per_scene: Dict[str, int] = defaultdict(lambda: 0)
        self.total_frames_per_scene: Dict[str, int] = defaultdict(lambda: -1)

    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        stage = input_stage
        stage = pypeln.thread.map(f=self.encode_scene, stage=stage, workers=1, maxsize=4)
        stage = pypeln.thread.map(f=self.encode_dataset, stage=stage, workers=1, maxsize=1)
        return stage

    def encode_scene(self, pipeline_item: TPipelineItem) -> TPipelineItem:
        if not pipeline_item.is_end_of_scene:
            self.seen_frames_per_scene[pipeline_item.scene_name] += 1
            self.on_end_of_sensor_frame(pipeline_item=pipeline_item)
        else:
            self.total_frames_per_scene[pipeline_item.scene_name] = pipeline_item.total_frames_in_scene

        total_frames = self.total_frames_per_scene[pipeline_item.scene_name]
        total_seen_frames = self.seen_frames_per_scene[pipeline_item.scene_name]
        if total_frames != -1 and total_seen_frames != -1 and total_seen_frames >= total_frames:
            self.on_end_of_scene(pipeline_item=pipeline_item)

        return pipeline_item

    def on_end_of_sensor_frame(self, pipeline_item: TPipelineItem):
        self.encoding_format.save_sensor_frame(pipeline_item=pipeline_item)

    def on_end_of_dataset(self, pipeline_item: TPipelineItem):
        self.encoding_format.save_dataset(pipeline_item=pipeline_item)

    def on_end_of_scene(self, pipeline_item: TPipelineItem):
        self.encoding_format.save_scene(pipeline_item=pipeline_item)

    def encode_dataset(self, pipeline_item: TPipelineItem) -> TPipelineItem:
        if pipeline_item.is_end_of_dataset:
            self.total_scenes = pipeline_item.total_scenes_in_dataset
        elif not pipeline_item.is_end_of_scene:
            self.total_seen_frames += 1
        else:
            if self.total_frames == -1:
                self.total_frames = 0
            self.total_frames += pipeline_item.total_frames_in_scene
            self.total_seen_scenes += 1

        seen_all_frames = (
            self.total_frames != -1 and self.total_seen_frames != -1 and self.total_seen_frames >= self.total_frames
        )
        seen_all_scenes = (
            self.total_scenes != -1 and self.total_seen_scenes != -1 and self.total_seen_scenes >= self.total_scenes
        )
        if seen_all_frames and seen_all_scenes:
            self.on_end_of_dataset(pipeline_item=pipeline_item)

        return pipeline_item


class GenericEncoderStep(Generic[TPipelineItem], EncoderStep):
    def __init__(
        self,
        encoding_format: EncodingFormat[TPipelineItem],
        copy_data_types: List[SensorDataTypes],
        fs_copy: bool,
        should_copy_callbacks: Optional[Dict[SensorDataTypes, Callable[[SensorDataTypes, SensorFrame], bool]]] = None,
        workers: int = 1,
        in_queue_size: int = 1,
    ):
        self.copy_data_types = copy_data_types
        self.should_copy_callbacks = should_copy_callbacks
        self.encoding_format = encoding_format
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.fs_copy = fs_copy

    def encode_frame_data(
        self,
        pipeline_item: TPipelineItem,
        data_type: SensorDataTypes,
        should_copy: Callable[[SensorDataTypes, SensorFrame], bool],
    ) -> TPipelineItem:
        sensor_frame = pipeline_item.sensor_frame

        if sensor_frame is not None and should_copy(data_type, sensor_frame):
            data_or_path = None

            load_data = True
            if self.fs_copy:
                if issubclass(data_type, Annotation) and data_type in sensor_frame.available_annotation_types:
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
                if issubclass(data_type, Annotation) and data_type in sensor_frame.available_annotation_types:
                    data_or_path = sensor_frame.get_annotations(annotation_type=data_type)
                elif issubclass(data_type, Image) and pipeline_item.camera_frame is not None:
                    data_or_path = pipeline_item.camera_frame.image.rgba
                elif issubclass(data_type, PointCloud) and pipeline_item.lidar_frame is not None:
                    data_or_path = pipeline_item.lidar_frame.point_cloud

            if data_or_path is not None:
                self.encoding_format.save_data(pipeline_item=pipeline_item, data=data_or_path, data_type=data_type)

        return pipeline_item

    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        for data_type in self.copy_data_types:

            def should_copy(d, f) -> bool:
                return True

            callback = should_copy
            if self.should_copy_callbacks is not None and data_type in self.should_copy_callbacks:
                callback = self.should_copy_callbacks[data_type]
            stage = pypeln.thread.map(
                f=partial(self.encode_frame_data, data_type=data_type, should_copy=callback),
                stage=stage,
                workers=self.workers,
                maxsize=self.in_queue_size,
            )
        return stage


class GenericPipelineBuilder(PipelineBuilder[TPipelineItem]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_format: str,
        output_path: AnyPath,
        pipeline_item_type: Type[TPipelineItem],
        scene_aggregator: EncoderStep = None,
        workers: int = 2,
        max_in_queue_size: int = 6,
        inplace: bool = False,
        custom_encoder_steps: List[EncoderStep] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        allowed_frames: Optional[List[FrameId]] = None,
        copy_data_types: Optional[List[SensorDataTypes]] = None,
        should_copy_callbacks: Optional[Dict[SensorDataTypes, Callable[[SensorDataTypes, SensorFrame], bool]]] = None,
        target_dataset_name: Optional[str] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        fs_copy: bool = True,
        copy_all_available_sensors_and_annotations: bool = False,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.pipeline_item_type = pipeline_item_type
        self.workers = workers
        self.max_in_queue_size = max_in_queue_size
        self.fs_copy = fs_copy
        self.inplace = inplace
        self.should_copy_callbacks = should_copy_callbacks
        self.target_dataset_name = target_dataset_name
        self.allowed_frames = allowed_frames
        self.output_path = output_path
        self.sensor_names = sensor_names

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        if "dataset_path" in decoder_kwargs:
            decoder_kwargs.pop("dataset_path")
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)

        if copy_data_types is None and copy_all_available_sensors_and_annotations:
            copy_data_types: List[SensorDataTypes] = dataset.available_annotation_types
            if len(dataset.camera_names) > 0:
                copy_data_types.append(FilePathedDataType.Image)
            if len(dataset.lidar_names) > 0:
                copy_data_types.append(FilePathedDataType.PointCloud)
        self.copy_data_types = copy_data_types

        self._dataset = dataset
        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

        self.scene_aggregator = scene_aggregator
        self.custom_encoder_steps = custom_encoder_steps

    def build_encoder_steps(self, encoding_format: EncodingFormat[TPipelineItem]) -> List[EncoderStep]:
        encoder_steps = list()
        if self.copy_data_types is not None:
            encoder_steps.append(
                GenericEncoderStep(
                    encoding_format=encoding_format,
                    copy_data_types=self.copy_data_types,
                    should_copy_callbacks=self.should_copy_callbacks,
                    fs_copy=self.fs_copy,
                    workers=self.workers,
                    in_queue_size=self.max_in_queue_size,
                )
            )
        if self.custom_encoder_steps is not None:
            for custom_step in self.custom_encoder_steps:
                encoder_steps.append(custom_step)

        if self.scene_aggregator is None:
            encoder_steps.append(GenericSceneAggregator(encoding_format=encoding_format))
        else:
            encoder_steps.append(self.scene_aggregator)
        return encoder_steps

    def build_pipeline_source_generator(self) -> Generator[TPipelineItem, None, None]:
        dataset = self._dataset
        for scene_name in self._scene_names:
            scene = self._dataset.get_unordered_scene(scene_name=scene_name)

            if self.sensor_names is None:
                sensor_name_mapping = {s: s for s in scene.sensor_names}
            elif isinstance(self.sensor_names, list):
                sensor_name_mapping = {s: s for s in self.sensor_names if s in scene.sensor_names}
            elif isinstance(self.sensor_names, dict):
                sensor_name_mapping = {t: s for t, s in self.sensor_names.items() if s in scene.sensor_names}
            else:
                raise ValueError(f"sensor_names is neither a list nor a dict but {type(self.sensor_names)}!")

            reference_timestamp: datetime = scene.get_frame(list(scene.frame_ids)[0]).date_time

            total_frames_in_scene = 0
            logger.info(f"Encoding Scene {scene.name} with sensor mapping: {sensor_name_mapping}")
            for target_sensor_name, source_sensor_name in sensor_name_mapping.items():
                sensor = scene.get_sensor(sensor_name=source_sensor_name)
                if sensor.name in sensor_name_mapping.values():
                    for frame_id in sensor.frame_ids:
                        if self.allowed_frames is None or frame_id in self.allowed_frames:
                            sensor_frame = sensor.get_frame(frame_id=frame_id)
                            yield self.pipeline_item_type(
                                sensor_name=sensor.name,
                                frame_id=sensor_frame.frame_id,
                                scene_name=scene.name,
                                dataset_path=dataset.path,
                                dataset_format=dataset.format,
                                decoder_kwargs=dataset.decoder_init_kwargs,
                                target_sensor_name=target_sensor_name,
                                scene_reference_timestamp=reference_timestamp,
                            )
                            total_frames_in_scene += 1

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
