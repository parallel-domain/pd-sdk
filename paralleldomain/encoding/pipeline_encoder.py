import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, Generic, Iterable, List, Optional, Type, TypeVar, Union

import pypeln
from pypeln.utils import A, BaseStage, Undefined
from tqdm import tqdm

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import (
    CameraSensor,
    CameraSensorFrame,
    LidarSensor,
    LidarSensorFrame,
    Sensor,
    SensorDataTypes,
    SensorFrame,
)
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger.addHandler(_TqdmLoggingHandler())


TSceneType = TypeVar("TSceneType", Scene, UnorderedScene)


@dataclass
class PipelineItem(Generic[TSceneType]):
    sensor_name: Optional[str]
    frame_id: Optional[str]
    scene_name: Optional[str]
    dataset_path: Union[str, AnyPath]
    dataset_path: Union[str, AnyPath]
    dataset_format: str
    decoder_kwargs: Dict[str, Any]
    target_sensor_name: Optional[str]
    scene_reference_timestamp: Optional[datetime]
    custom_data: Dict[str, Any] = field(default_factory=dict)
    is_end_of_scene: bool = False
    is_end_of_dataset: bool = False
    total_frames_in_scene: Optional[int] = -1
    total_scenes_in_dataset: Optional[int] = -1

    @property
    def dataset(self) -> Dataset:
        return decode_dataset(dataset_format=self.dataset_format, **self.decoder_kwargs)

    @property
    @abstractmethod
    def scene(self) -> Optional[TSceneType]:
        pass

    @property
    def sensor(self) -> Optional[Sensor]:
        if self.sensor_name is not None:
            return self.scene.get_sensor(sensor_name=self.sensor_name)
        return None

    @property
    def sensor_frame(self) -> Optional[SensorFrame]:
        if self.frame_id is not None:
            return self.sensor.get_frame(frame_id=self.frame_id)
        return None

    @property
    def camera_frame(self) -> Optional[CameraSensorFrame]:
        if self.sensor is not None and isinstance(self.sensor, CameraSensor):
            return self.sensor.get_frame(frame_id=self.frame_id)
        return None

    @property
    def lidar_frame(self) -> Optional[LidarSensorFrame]:
        if self.sensor is not None and isinstance(self.sensor, LidarSensor):
            return self.sensor.get_frame(frame_id=self.frame_id)
        return None


@dataclass
class ScenePipelineItem(PipelineItem[Scene]):
    @property
    def scene(self) -> Optional[Scene]:
        if self.scene_name is not None:
            return self.dataset.get_scene(scene_name=self.scene_name)
        return None


@dataclass
class UnorderedScenePipelineItem(PipelineItem[UnorderedScene]):
    @property
    def scene(self) -> Optional[UnorderedScene]:
        if self.scene_name is not None:
            return self.dataset.get_unordered_scene(scene_name=self.scene_name)
        return None


TPipelineItem = TypeVar("TPipelineItem", PipelineItem, Dict)
DataType = Union[SensorDataTypes, Type[ClassMap]]


class EncodingFormat(Generic[TPipelineItem]):
    @abstractmethod
    def save_data(self, pipeline_item: TPipelineItem, data_type: DataType, data: Any):
        pass

    @abstractmethod
    def supports_copy(self, pipeline_item: TPipelineItem, data_type: DataType, data_path: AnyPath):
        pass

    @abstractmethod
    def save_sensor_frame(self, pipeline_item: TPipelineItem, data: Any = None):
        pass

    @abstractmethod
    def save_scene(self, pipeline_item: TPipelineItem, data: Any = None):
        pass

    @abstractmethod
    def save_dataset(self, pipeline_item: TPipelineItem, data: Any = None):
        pass


class EncoderStep:
    @abstractmethod
    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass


class PipelineBuilder(Generic[TPipelineItem]):
    @abstractmethod
    def build_encoder_steps(self, encoding_format: EncodingFormat[TPipelineItem]) -> List[EncoderStep]:
        pass

    @abstractmethod
    def build_pipeline_source_generator(self) -> Generator[TPipelineItem, None, None]:
        pass

    @property
    @abstractmethod
    def pipeline_item_unit_name(self):
        pass


class DatasetPipelineEncoder(Generic[TPipelineItem]):
    def __init__(
        self,
        pipeline_builder: PipelineBuilder[TPipelineItem],
        encoding_format: EncodingFormat[TPipelineItem],
        use_tqdm: bool = True,
    ):
        self.encoding_format = encoding_format
        self.pipeline_builder = pipeline_builder
        self.use_tqdm = use_tqdm

    @staticmethod
    def build_pipeline(
        source_generator: Generator[TPipelineItem, None, None], encoder_steps: List[EncoderStep]
    ) -> Union[BaseStage[A], Iterable[A], Undefined]:
        stage = source_generator
        for encoder in encoder_steps:
            stage = encoder.apply(input_stage=stage)
        return stage

    def encode_dataset(self):
        stage = self.pipeline_builder.build_pipeline_source_generator()
        encoder_steps = self.pipeline_builder.build_encoder_steps(encoding_format=self.encoding_format)
        stage = self.build_pipeline(source_generator=stage, encoder_steps=encoder_steps)

        stage = pypeln.process.to_iterable(stage)
        if self.use_tqdm:
            stage = tqdm(
                stage,
                desc="Encoding Progress",
                unit=f" {self.pipeline_builder.pipeline_item_unit_name}",
                smoothing=0.0,
            )
        for _ in stage:
            pass

    @classmethod
    def from_builder(
        cls,
        pipeline_builder: PipelineBuilder,
        encoding_format: EncodingFormat,
        use_tqdm: bool = True,
        **kwargs,
    ) -> "DatasetPipelineEncoder":
        return cls(use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format)
