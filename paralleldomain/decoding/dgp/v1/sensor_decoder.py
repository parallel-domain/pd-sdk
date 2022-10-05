import abc
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Set, Type

from paralleldomain.common.dgp.v1 import sample_pb2
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.v1.sensor_frame_decoder import (
    DGPCameraSensorFrameDecoder,
    DGPLidarSensorFrameDecoder,
    DGPRadarSensorFrameDecoder,
)
from paralleldomain.decoding.sensor_decoder import (
    CameraSensorDecoder,
    LidarSensorDecoder,
    RadarSensorDecoder,
    SensorDecoder,
)
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.annotation import Annotation
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


class DGPSensorDecoder(SensorDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, sample_pb2.Sample],
        scene_data: List[sample_pb2.Datum],
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        settings: DecoderSettings,
        annotation_type_map: Dict[str, Type[Annotation]],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.scene_data = scene_data
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.scene_samples = scene_samples
        self.dataset_path = dataset_path
        self._ontologies = ontologies
        self._annotation_type_map = annotation_type_map
        self._data_by_key = lru_cache(maxsize=1)(self._data_by_key)

    def _data_by_key(self) -> Dict[str, sample_pb2.Datum]:
        return {d.key: d for d in self.scene_data}

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        sensor_data = self._data_by_key()

        frame_ids = list(self.scene_samples.keys())
        sensor_frame_ids = set()
        for frame_id in frame_ids:
            sample = self.scene_samples[frame_id]
            for key in sample.datum_keys:
                if sensor_data[key].id.name == sensor_name:
                    sensor_frame_ids.add(frame_id)
                    break

        return sensor_frame_ids


class DGPCameraSensorDecoder(DGPSensorDecoder, CameraSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            annotation_type_map=self._annotation_type_map,
        )


class DGPLidarSensorDecoder(DGPSensorDecoder, LidarSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_lidar_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_lidar_sensor_frame_decoder)

    def _create_lidar_sensor_frame_decoder(self) -> DGPLidarSensorFrameDecoder:
        return DGPLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            annotation_type_map=self._annotation_type_map,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, lidar_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        return LidarSensorFrame[datetime](sensor_name=lidar_name, frame_id=frame_id, decoder=decoder)


class DGPRadarSensorDecoder(DGPSensorDecoder, RadarSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_radar_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_radar_sensor_frame_decoder)

    def _create_radar_sensor_frame_decoder(self) -> DGPRadarSensorFrameDecoder:
        return DGPRadarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            annotation_type_map=self._annotation_type_map,
        )

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[datetime], frame_id: FrameId, radar_name: SensorName
    ) -> RadarSensorFrame[datetime]:
        return RadarSensorFrame[datetime](sensor_name=radar_name, frame_id=frame_id, decoder=decoder)
