import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from google.protobuf.json_format import MessageToDict

from paralleldomain.common.dgp.v1 import dataset_pb2, ontology_pb2, sample_pb2, scene_pb2
from paralleldomain.common.dgp.v1.common import timestamp_to_datetime
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP
from paralleldomain.common.dgp.v1.metadata_pd_pb2 import ParallelDomainSceneMetadata
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, FrameDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.dgp.v1.frame_decoder import DGPFrameDecoder
from paralleldomain.decoding.dgp.v1.sensor_decoder import DGPCameraSensorDecoder, DGPLidarSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json_message
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


class _DatasetDecoderMixin:
    def __init__(self, dataset_path: Union[str, AnyPath], **kwargs):
        self._dataset_path: AnyPath = AnyPath(dataset_path)

    def _decode_scene_paths(self) -> List[Path]:
        dto = self._decode_dataset_dto()
        return [
            Path(path)
            for split_key in sorted(dto.scene_splits.keys())
            for path in dto.scene_splits[split_key].filenames
        ]

    @lru_cache(maxsize=1)
    def _decode_dataset_dto(self) -> dataset_pb2.SceneDataset:
        dataset_cloud_path: AnyPath = self._dataset_path
        scene_dataset_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_dataset_json_path.exists():
            raise FileNotFoundError(f"File {scene_dataset_json_path} not found.")

        scene_dataset_dto = read_json_message(obj=dataset_pb2.SceneDataset(), path=scene_dataset_json_path)

        return scene_dataset_dto

    def _decode_scene_names(self) -> List[SceneName]:
        return [p.parent.name for p in self._decode_scene_paths()]


class DGPDatasetDecoder(_DatasetDecoderMixin, DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        custom_reference_to_box_bottom: Optional[Transformation] = None,
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        _DatasetDecoderMixin.__init__(self, dataset_path=dataset_path)
        DatasetDecoder.__init__(self, dataset_name=str(dataset_path), settings=settings)
        self.custom_reference_to_box_bottom = (
            Transformation() if custom_reference_to_box_bottom is None else custom_reference_to_box_bottom
        )

        self._dataset_path: AnyPath = AnyPath(dataset_path)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return DGPSceneDecoder(
            dataset_path=self._dataset_path,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [p.parent.name for p in self._decode_scene_paths()]

    def _decode_dataset_metadata(self) -> DatasetMeta:
        dto = self._decode_dataset_dto()
        anno_types = [ANNOTATION_TYPE_MAP[a] for a in dto.metadata.available_annotation_types]
        return DatasetMeta(
            name=dto.metadata.name, available_annotation_types=anno_types, custom_attributes=dto.metadata
        )


class DGPSceneDecoder(SceneDecoder[datetime], _DatasetDecoderMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: DecoderSettings,
        custom_reference_to_box_bottom: Optional[Transformation] = None,
    ):
        _DatasetDecoderMixin.__init__(self, dataset_path=dataset_path)
        SceneDecoder.__init__(self, dataset_name=str(dataset_path), settings=settings)

        self.custom_reference_to_box_bottom = (
            Transformation() if custom_reference_to_box_bottom is None else custom_reference_to_box_bottom
        )

        self._dataset_path: AnyPath = AnyPath(dataset_path)

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[TDateTime]:
        return DGPCameraSensorDecoder(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self._sample_by_index(scene_name=scene_name),
            scene_data=self._decode_scene_dto(scene_name=scene_name).data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[TDateTime]:
        return DGPLidarSensorDecoder(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self._sample_by_index(scene_name=scene_name),
            scene_data=self._decode_scene_dto(scene_name=scene_name).data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        return {str(sample.id.index): timestamp_to_datetime(sample.id.timestamp) for sample in scene_dto.samples}

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        metadata = {k: v for k, v in scene_dto.metadata.items()}
        if "PD" in metadata:
            metadata["PD"] = dict(MessageToDict(scene_dto.metadata["PD"], ParallelDomainSceneMetadata()))
        return metadata

    def _decode_set_description(self, scene_name: SceneName) -> str:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        return scene_dto.description

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        return set(self._decode_frame_id_to_date_time_map(scene_name=scene_name).keys())

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data}))

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.HasField("image")}))

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.HasField("point_cloud")}))

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        scene_dto = self._decode_scene_dto(scene_name=scene_name)

        ontologies = {}
        for annotation_key, ontology_file in scene_dto.ontologies.items():
            ontology_path = self._dataset_path / scene_name / "ontology" / f"{ontology_file}.json"
            ontology_dto = read_json_message(obj=ontology_pb2.Ontology(), path=ontology_path)

            ontologies[ANNOTATION_TYPE_MAP[annotation_key]] = ClassMap(
                classes=[
                    ClassDetail(
                        name=o.name,
                        id=o.id,
                        instanced=o.isthing,
                        meta={"color": {"r": o.color.r, "g": o.color.g, "b": o.color.b}},
                    )
                    for o in ontology_dto.items
                ]
            )

        return ontologies

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder:
        return DGPFrameDecoder(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            scene_samples=self._sample_by_index(scene_name=scene_name),
            scene_data=self._decode_scene_dto(scene_name=scene_name).data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    @lru_cache(maxsize=1)
    def _decode_scene_dto(self, scene_name: str) -> scene_pb2.Scene:
        scene_names = self._decode_scene_names()
        scene_index = scene_names.index(scene_name)

        scene_paths = self._decode_scene_paths()
        scene_path = scene_paths[scene_index]

        scene_file = self._dataset_path / scene_path

        scene_dto = read_json_message(obj=scene_pb2.Scene(), path=scene_file)

        return scene_dto

    @lru_cache(maxsize=1)
    def _data_by_key_with_name(self, scene_name: str, data_name: str) -> Dict[str, sample_pb2.Datum]:
        dto = self._decode_scene_dto(scene_name=scene_name)
        return {d.key: d for d in dto.data if d.id.name == data_name}

    @lru_cache(maxsize=1)
    def _sample_by_index(self, scene_name: str) -> Dict[str, sample_pb2.Sample]:
        dto = self._decode_scene_dto(scene_name=scene_name)
        return {str(s.id.index): s for s in dto.samples}
