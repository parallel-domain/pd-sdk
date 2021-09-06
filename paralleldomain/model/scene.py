from datetime import datetime
from typing import List, TypeVar

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor_frame_set import SensorFrameSet, SensorFrameSetDecoderProtocol

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame
from paralleldomain.model.type_aliases import FrameId, SensorFrameSetName

T = TypeVar("T")


class Scene(SensorFrameSet[datetime]):
    """A collection of time-ordered sensor data.

    Args:
        name: Name of scene
        available_annotation_types: List of available annotation types for this scene.
        decoder: Decoder instance to be used for loading all relevant objects (frames, annotations etc.)
    """

    def __init__(
        self,
        name: SensorFrameSetName,
        available_annotation_types: List[AnnotationType],
        decoder: SensorFrameSetDecoderProtocol[datetime],
    ):
        super().__init__(name=name, available_annotation_types=available_annotation_types, decoder=decoder)
        self._decoder = decoder

    @property
    def frames(self) -> List[Frame]:
        return [self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids]

    @property
    def frame_ids(self) -> List[FrameId]:
        fids = list(self._decoder.get_frame_ids(set_name=self.name))
        return sorted(fids, key=self._decoder.get_frame_id_to_date_time_map(scene_name=self.name).get)
