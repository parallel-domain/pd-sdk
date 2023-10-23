import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import iso8601

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.common.dgp.v0.dtos import SceneDataDTO, SceneDTO, SceneSampleDTO
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import FrameDecoder, SceneDecoder
from paralleldomain.decoding.dgp.common import _DatasetDecoderMixin, decode_class_maps
from paralleldomain.decoding.dgp.frame_decoder import DGPFrameDecoder
from paralleldomain.decoding.dgp.sensor_decoder import DGPCameraSensorDecoder, DGPLidarSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


T = TypeVar("T")


class DGPSceneDecoder(SceneDecoder[datetime], _DatasetDecoderMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        scene_name: SceneName,
        settings: DecoderSettings,
        custom_reference_to_box_bottom: Optional[Transformation] = None,
    ):
        _DatasetDecoderMixin.__init__(self, dataset_path=dataset_path)
        SceneDecoder.__init__(self, dataset_name=str(dataset_path), settings=settings, scene_name=scene_name)
        self.custom_reference_to_box_bottom = (
            Transformation() if custom_reference_to_box_bottom is None else custom_reference_to_box_bottom
        )

        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self._scene_dto = None
        self._sample_by_index = None
        point_cache_folder = self._dataset_path / scene_name / "point_cache"
        self._point_cache_folder_exists = point_cache_folder.exists()

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[datetime]:
        scene_dto = self.scene_dto
        return DGPCameraSensorDecoder(
            dataset_name=self.dataset_name,
            sensor_name=sensor_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self.sample_by_index,
            scene_data=scene_dto.data,
            ontologies=scene_dto.ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=False,
            point_cache_folder_exists=self._point_cache_folder_exists,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[datetime]:
        scene_dto = self.scene_dto
        return DGPLidarSensorDecoder(
            dataset_name=self.dataset_name,
            sensor_name=sensor_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self.sample_by_index,
            scene_data=scene_dto.data,
            ontologies=scene_dto.ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=False,
            point_cache_folder_exists=self._point_cache_folder_exists,
        )

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        dto = self._decode_dataset_dto()
        return [
            AnnotationIdentifier(annotation_type=ANNOTATION_TYPE_MAP[str(a)])
            for a in dto.metadata.available_annotation_types
        ]

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        scene_dto = self.scene_dto
        return {sample.id.index: self._scene_sample_to_date_time(sample=sample) for sample in scene_dto.samples}

    def _decode_set_metadata(self) -> Dict[str, Any]:
        scene_dto = self.scene_dto
        return scene_dto.metadata

    def _decode_set_description(self) -> str:
        scene_dto = self.scene_dto
        return scene_dto.description

    def _decode_frame_id_set(self) -> Set[FrameId]:
        return set(self._decode_frame_id_to_date_time_map().keys())

    def _decode_sensor_names(self) -> List[SensorName]:
        scene_dto = self.scene_dto
        return sorted(list({datum.id.name for datum in scene_dto.data}))

    def _decode_camera_names(self) -> List[SensorName]:
        scene_dto = self.scene_dto
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.image}))

    def _decode_lidar_names(self) -> List[SensorName]:
        scene_dto = self.scene_dto
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.point_cloud}))

    def _decode_radar_names(self) -> List[SensorName]:
        """Radar not supported in v0"""
        return list()

    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[datetime]:
        raise ValueError("DGP V0 does not support radar data!")

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        scene_dto = self.scene_dto
        return decode_class_maps(
            ontologies=scene_dto.ontologies, dataset_path=self._dataset_path, scene_name=self.scene_name
        )

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[datetime]:
        scene_dto = self.scene_dto
        return DGPFrameDecoder(
            dataset_name=self.dataset_name,
            frame_id=frame_id,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self.sample_by_index,
            scene_data=scene_dto.data,
            ontologies=scene_dto.ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=False,
            point_cache_folder_exists=self._point_cache_folder_exists,
        )

    @staticmethod
    def _scene_sample_to_date_time(sample: SceneSampleDTO) -> datetime:
        return iso8601.parse_date(sample.id.timestamp)

    @property
    def scene_dto(self) -> SceneDTO:
        if self._scene_dto is None:
            scene_names = self._decode_scene_names()
            scene_index = scene_names.index(self.scene_name)

            scene_paths = self._decode_scene_paths()
            scene_path = scene_paths[scene_index]

            scene_file = self._dataset_path / scene_path

            scene_data = read_json(path=scene_file)

            scene_dto = SceneDTO.from_dict(scene_data)
            self._scene_dto = scene_dto
        return self._scene_dto

    @property
    def sample_by_index(self) -> Dict[str, SceneSampleDTO]:
        if self._sample_by_index is None:
            self._sample_by_index = {s.id.index: s for s in self.scene_dto.samples}
        return self._sample_by_index
