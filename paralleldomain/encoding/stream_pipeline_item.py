from typing import Any, Dict, List, Optional

from paralleldomain import Dataset, Scene
from paralleldomain.encoding.pipeline_encoder import ScenePipelineItem
from paralleldomain.model.dataset import DatasetDecoderProtocol
from paralleldomain.model.frame import Frame, FrameDecoderProtocol
from paralleldomain.model.scene import SceneDecoderProtocol
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, Sensor, SensorFrame


class StreamPipelineItem(ScenePipelineItem):
    def __init__(
        self,
        frame_decoder: Optional[FrameDecoderProtocol],
        scene_decoder: Optional[SceneDecoderProtocol],
        dataset_decoder: DatasetDecoderProtocol,
        available_annotation_types: List,
        custom_data: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_decoder = frame_decoder
        self.scene_decoder = scene_decoder
        self.available_annotation_types = available_annotation_types
        self.dataset_decoder = dataset_decoder
        self.custom_data = custom_data if custom_data is not None else dict()

    @property
    def dataset(self) -> Dataset:
        return Dataset(decoder=self.dataset_decoder)

    @property
    def scene(self) -> Optional[Scene]:
        if self.scene_decoder is not None:
            return Scene(
                decoder=self.scene_decoder,
                name=self.scene_name,
                available_annotation_types=self.available_annotation_types,
            )
        return None

    @property
    def sensor(self) -> Optional[Sensor]:
        return None

    @property
    def frame(self) -> Optional[Frame]:
        if self.frame_decoder is not None:
            return Frame[None](
                frame_id=self.frame_id,
                decoder=self.frame_decoder,
            )
        return None

    @property
    def sensor_frame(self) -> Optional[SensorFrame]:
        if self.frame is not None and self.sensor_name is not None:
            return self.frame.get_sensor(sensor_name=self.sensor_name)
        return None

    @property
    def camera_frame(self) -> Optional[CameraSensorFrame]:
        if self.frame is not None and self.sensor_name in self.frame.camera_names:
            return self.frame.get_camera(camera_name=self.sensor_name)
        return None

    @property
    def lidar_frame(self) -> Optional[LidarSensorFrame]:
        if self.frame is not None and self.sensor_name in self.frame.lidar_names:
            return self.frame.get_lidar(lidar_name=self.sensor_name)
        return None
