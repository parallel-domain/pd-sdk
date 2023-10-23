import abc
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Set

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
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


class DGPSensorDecoder(SensorDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, sample_pb2.Sample],
        scene_data: List[sample_pb2.Datum],
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        point_cache_folder_exists: bool,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            is_unordered_scene=is_unordered_scene,
            scene_decoder=scene_decoder,
        )
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.dataset_path = dataset_path
        self._point_cache_folder_exists = point_cache_folder_exists
        self._ontologies = ontologies

        self.fid_to_sensor_data = dict()
        self.key_to_sensor_data = {d.key: d for d in scene_data if d.id.name == sensor_name}
        self.sensor_data_keys = {d.key for d in self.key_to_sensor_data.values()}
        self.frame_samples = {
            fid: s for fid, s in scene_samples.items() if any([k in self.sensor_data_keys for k in s.datum_keys])
        }
        for fid, s in self.frame_samples.items():
            for key in s.datum_keys:
                if key in self.key_to_sensor_data:
                    self.fid_to_sensor_data[fid] = self.key_to_sensor_data[key]

    def _decode_frame_id_set(self) -> Set[FrameId]:
        return set(self.frame_samples.keys())


class DGPCameraSensorDecoder(DGPSensorDecoder, CameraSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            frame_sample=self.frame_samples[frame_id],
            sensor_frame_data=self.fid_to_sensor_data[frame_id],
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )


class DGPLidarSensorDecoder(DGPSensorDecoder, LidarSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> DGPLidarSensorFrameDecoder:
        return DGPLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            frame_sample=self.frame_samples[frame_id],
            sensor_frame_data=self.fid_to_sensor_data[frame_id],
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )


class DGPRadarSensorDecoder(DGPSensorDecoder, RadarSensorDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_radar_sensor_frame_decoder(self, frame_id: FrameId) -> DGPRadarSensorFrameDecoder:
        return DGPRadarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            frame_sample=self.frame_samples[frame_id],
            sensor_frame_data=self.fid_to_sensor_data[frame_id],
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            ontologies=self._ontologies,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
