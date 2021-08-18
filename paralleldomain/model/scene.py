import contextlib
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, List, Type, TypeVar, cast

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.ego import EgoFrame
from paralleldomain.model.sensor_frame_set import SensorFrameSet, SensorFrameSetDecoderProtocol
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame, TemporalFrame
from paralleldomain.model.sensor import CameraSensor, LidarSensor, Sensor, TemporalSensorFrame
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorFrameSetName, SensorName

T = TypeVar("T")


class SceneDecoderProtocol(SensorFrameSetDecoderProtocol, Protocol):
    def get_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        pass


class Scene(SensorFrameSet[TemporalFrame, TemporalSensorFrame]):
    def __init__(
        self,
        name: SensorFrameSetName,
        available_annotation_types: List[AnnotationType],
        decoder: SceneDecoderProtocol,
    ):
        super().__init__(name=name, available_annotation_types=available_annotation_types, decoder=decoder)
        self._decoder = decoder

    @property
    def ordered_frame_ids(self) -> List[FrameId]:
        return sorted(self.frame_ids, key=self._decoder.get_frame_id_to_date_time_map(scene_name=self.name).get)

    @classmethod
    def from_decoder(
        cls, scene_name: SceneName, available_annotation_types: List[AnnotationType], decoder: SceneDecoderProtocol
    ) -> "Scene":
        return cls(
            name=scene_name,
            available_annotation_types=available_annotation_types,
            decoder=decoder,
        )
