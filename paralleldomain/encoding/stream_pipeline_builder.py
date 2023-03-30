import logging
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union

import pypeln

from paralleldomain import Scene
from paralleldomain.common.constants import ANNOTATION_NAME_TO_CLASS
from paralleldomain.data_lab.config.sensor_rig import SensorRig
from paralleldomain.decoding.in_memory.dataset_decoder import InMemoryDatasetDecoder
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.step.constants import PD_CLASS_DETAILS
from paralleldomain.encoding.generic_pipeline_builder import GenericEncoderStep, GenericSceneAggregator
from paralleldomain.encoding.pipeline_encoder import (
    DatasetPipelineEncoder,
    EncoderStep,
    EncodingFormat,
    PipelineBuilder,
)
from paralleldomain.encoding.stream_pipeline_item import StreamPipelineItem
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import FilePathedDataType, SensorDataTypes, SensorFrame
from paralleldomain.model.type_aliases import FrameId
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class StreamGenericEncoderStep(GenericEncoderStep):
    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        for data_type in self.copy_data_types:

            def should_copy(d, f) -> bool:
                return True

            callback = should_copy
            if self.should_copy_callbacks is not None and data_type in self.should_copy_callbacks:
                callback = self.should_copy_callbacks[data_type]
            stage = self.run_env.map(
                f=partial(self.encode_frame_data, data_type=data_type, should_copy=callback),
                stage=stage,
                workers=self.workers,
                maxsize=self.in_queue_size,
            )
        return stage


class StreamEncodingPipelineBuilder(PipelineBuilder[StreamPipelineItem]):
    def __init__(
        self,
        frame_stream: Generator[Tuple[Frame, Scene], None, None],
        sensor_rig: SensorRig,
        number_of_scenes: int,
        number_of_frames_per_scene: int,
        scene_aggregator: EncoderStep = None,
        workers: int = 4,
        max_in_queue_size: int = 6,
        custom_encoder_steps: List[EncoderStep] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        allowed_frames: Optional[List[FrameId]] = None,
        copy_data_types: Optional[List[SensorDataTypes]] = None,
        should_copy_callbacks: Optional[Dict[SensorDataTypes, Callable[[SensorDataTypes, SensorFrame], bool]]] = None,
        target_dataset_name: str = "MiniBatchDataset",
        copy_all_available_sensors_and_annotations: bool = False,
        run_env: Literal["thread", "process", "sync"] = "thread",
    ):
        self.number_of_scenes = number_of_scenes
        self.number_of_frames_per_scene = number_of_frames_per_scene
        self._seen_scene_frames = dict()
        self.workers = workers
        self.sensor_rig = sensor_rig
        self.frame_stream = frame_stream
        self.max_in_queue_size = max_in_queue_size
        self.should_copy_callbacks = should_copy_callbacks
        self.target_dataset_name = target_dataset_name
        self.allowed_frames = allowed_frames
        self.sensor_names = sensor_names
        self.run_env = run_env

        if copy_data_types is None and copy_all_available_sensors_and_annotations:
            copy_data_types: List[Union[SensorDataTypes, AnnotationType]] = sensor_rig.available_annotations
            if len(sensor_rig.cameras) > 0:
                copy_data_types.append(FilePathedDataType.Image)
            if len(sensor_rig.lidars) > 0:
                copy_data_types.append(FilePathedDataType.PointCloud)
        self.copy_data_types = copy_data_types

        self.scene_aggregator = scene_aggregator
        self.custom_encoder_steps = custom_encoder_steps

    def build_encoder_steps(self, encoding_format: EncodingFormat[StreamPipelineItem]) -> List[EncoderStep]:
        encoder_steps = list()
        if self.copy_data_types is not None:
            encoder_steps.append(
                StreamGenericEncoderStep(
                    encoding_format=encoding_format,
                    copy_data_types=self.copy_data_types,
                    should_copy_callbacks=self.should_copy_callbacks,
                    fs_copy=False,
                    workers=self.workers,
                    in_queue_size=self.max_in_queue_size,
                    run_env=self.run_env,
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

    def build_pipeline_source_generator(self) -> Generator[StreamPipelineItem, None, None]:
        total_frames_in_scene = 0
        logger.info("Encoding Scene Step Stream")
        if self.sensor_names is None:
            sensor_name_mapping = {s: s for s in self.sensor_rig.sensor_names}
        elif isinstance(self.sensor_names, list):
            sensor_name_mapping = {s: s for s in self.sensor_names if s in self.sensor_rig.sensor_names}
        elif isinstance(self.sensor_names, dict):
            sensor_name_mapping = {t: s for t, s in self.sensor_names.items() if s in self.sensor_rig.sensor_names}
        else:
            raise ValueError(f"sensor_names is neither a list nor a dict but {type(self.sensor_names)}!")
        scene_reference_timestamp = datetime.min

        dataset_decoder = InMemoryDatasetDecoder(
            scene_names=[],
            unordered_scene_names=[],
            metadata=DatasetMeta(
                name=self.target_dataset_name, available_annotation_types=self.sensor_rig.available_annotations
            ),
        )
        scene_decoders = dict()

        for frame, scene in self.frame_stream:
            scene_name = frame.scene_name
            if scene_name not in dataset_decoder.scene_names:
                dataset_decoder.scene_names.append(scene_name)

            in_memory_frame_decoder = InMemoryFrameDecoder.from_frame(frame=frame)

            if scene_name not in self._seen_scene_frames:
                self._seen_scene_frames[scene_name] = 0

                scene_decoder = InMemorySceneDecoder.from_scene(scene)
                scene_decoders[scene_name] = scene_decoder

            self._seen_scene_frames[scene_name] += 1

            for sensor_name in frame.sensor_names:
                yield StreamPipelineItem(
                    frame_decoder=in_memory_frame_decoder,
                    sensor_name=sensor_name,
                    frame_id=frame.frame_id,
                    scene_name=scene_name,
                    dataset_path=None,
                    dataset_format="step",
                    decoder_kwargs=dict(),
                    target_sensor_name=sensor_name_mapping[sensor_name],
                    scene_reference_timestamp=scene_reference_timestamp,
                    dataset_decoder=dataset_decoder,
                    scene_decoder=scene_decoders[scene_name],
                    available_annotation_types=self.sensor_rig.available_annotations,
                )
                total_frames_in_scene += 1

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
                    total_frames_in_scene=self.number_of_frames_per_scene,
                    frame_decoder=None,
                    dataset_decoder=dataset_decoder,
                    scene_decoder=scene_decoders[scene_name],
                    available_annotation_types=self.sensor_rig.available_annotations,
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
            total_scenes_in_dataset=self.number_of_scenes,
            frame_decoder=None,
            dataset_decoder=dataset_decoder,
            scene_decoder=None,
            available_annotation_types=self.sensor_rig.available_annotations,
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
