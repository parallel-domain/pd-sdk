import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import PosixPath
from typing import Any, Callable, Dict, List, Optional, Union

import iso8601

from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp.constants import ANNOTATION_TYPE_MAP
from paralleldomain.decoding.dgp.dtos import DatasetDTO, OntologyFileDTO, SceneDataDTO, SceneDTO, SceneSampleDTO
from paralleldomain.decoding.dgp.frame_lazy_loader import DGPFrameLazyLoader
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.sensor import CameraSensor, LidarSensor, Sensor, SensorFrame
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPDecoder(Decoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        max_scene_dtos_to_cache: int = 10,
        custom_reference_to_box_bottom: Optional[Transformation] = None,
    ):
        self.custom_reference_to_box_bottom = (
            Transformation() if custom_reference_to_box_bottom is None else custom_reference_to_box_bottom
        )

        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self.decode_scene = lru_cache(max_scene_dtos_to_cache)(self.decode_scene)

    @lru_cache(maxsize=1)
    def _data_by_key(self, scene_name: str) -> Dict[str, SceneDataDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {d.key: d for d in dto.data}

    @lru_cache(maxsize=1)
    def _data_by_key_with_name(self, scene_name: str, data_name: str) -> Dict[str, SceneDataDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {d.key: d for d in dto.data if d.id.name == data_name}

    @lru_cache(maxsize=1)
    def _sample_by_index(self, scene_name: str) -> Dict[str, SceneSampleDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {s.id.index: s for s in dto.samples}

    @lru_cache(maxsize=1)
    def decode_dataset(self) -> DatasetDTO:
        dataset_cloud_path: AnyPath = self._dataset_path
        scene_dataset_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_dataset_json_path.exists():
            raise FileNotFoundError(f"File {scene_dataset_json_path} not found.")

        with scene_dataset_json_path.open(mode="r") as f:
            scene_dataset_json = json.load(f)
        scene_dataset_dto = DatasetDTO.from_dict(scene_dataset_json)

        return scene_dataset_dto

    def decode_scene(self, scene_name: str) -> SceneDTO:
        scene_names = self.decode_scene_names()
        scene_index = scene_names.index(scene_name)

        scene_paths = self.decode_scene_paths()
        scene_path = scene_paths[scene_index]

        scene_file = self._dataset_path / scene_path

        with scene_file.open("r") as f:
            scene_data = json.load(f)

        scene_dto = SceneDTO.from_dict(scene_data)
        return scene_dto

    # ------------------------------------------------
    def get_unique_scene_id(self, scene_name: SceneName) -> str:
        return f"{self._dataset_path}-{scene_name}"

    def decode_scene_names(self) -> List[str]:
        return [p.parent.name for p in self.decode_scene_paths()]

    def decode_scene_paths(self) -> List[PosixPath]:
        dto = self.decode_dataset()
        return [
            PosixPath(path)
            for split_key in sorted(dto.scene_splits.keys())
            for path in dto.scene_splits[split_key].filenames
        ]

    def decode_dataset_meta_data(self) -> DatasetMeta:
        dto = self.decode_dataset()
        meta_dict = dto.metadata.to_dict()
        anno_types = [ANNOTATION_TYPE_MAP[str(a)] for a in dto.metadata.available_annotation_types]
        return DatasetMeta(name=dto.metadata.name, available_annotation_types=anno_types, custom_attributes=meta_dict)

    def decode_scene_description(self, scene_name: SceneName) -> str:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return scene_dto.description

    def decode_scene_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return scene_dto.metadata.to_dict()

    def decode_class_maps(self, scene_name: SceneName) -> Dict[str, ClassMap]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        ontologies = {}
        for annotation_key, ontology_file in scene_dto.ontologies.items():
            with (self._dataset_path / scene_name / "ontology" / f"{ontology_file}.json").open() as fp:
                ontology_data = json.load(fp)

            ontology_dto = OntologyFileDTO.from_dict(ontology_data)
            ontologies[annotation_key] = ClassMap(
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

    def decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return {sample.id.index: self._scene_sample_to_date_time(sample=sample) for sample in scene_dto.samples}

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data}))

    def decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.image}))

    def decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return sorted(list({datum.id.name for datum in scene_dto.data if datum.datum.point_cloud}))

    def decode_sensor(
        self,
        scene_name: SceneName,
        sensor_name: SensorName,
        sensor_frame_factory: Callable[[FrameId, SensorName], SensorFrame],
    ) -> Sensor:
        sensor_data = self._data_by_key_with_name(scene_name=scene_name, data_name=sensor_name)
        data = next(iter(sensor_data.values()))

        if data.datum.point_cloud:
            return LidarSensor(sensor_name=sensor_name, sensor_frame_factory=sensor_frame_factory)
        elif data.datum.image:
            return CameraSensor(sensor_name=sensor_name, sensor_frame_factory=sensor_frame_factory)

        raise ValueError(f"Unknown Sensor type {sensor_name}!")

    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        # sample of current frame
        sample = self._sample_by_index(scene_name=scene_name)[frame_id]
        # all sensor data of the sensor
        sensor_data = self._data_by_key(scene_name=scene_name)
        return [sensor_data[key].id.name for key in sample.datum_keys]

    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        # sample of current frame
        sample = self._sample_by_index(scene_name=scene_name)[frame_id]
        # all sensor data of the sensor
        sensor_data = self._data_by_key_with_name(scene_name=scene_name, data_name=sensor_name)
        # read ontology -> Dict[str, ClassMap]
        # datum ley of sample that references the given sensor name
        datum_key = next(iter([key for key in sample.datum_keys if key in sensor_data]))
        scene_data = sensor_data[datum_key]
        unique_cache_key = f"{self._dataset_path}-{scene_name}-{frame_id}-{sensor_name}"
        sensor_frame = SensorFrame(
            unique_cache_key=unique_cache_key,
            frame_id=frame_id,
            date_time=self._scene_sample_to_date_time(sample=sample),
            sensor_name=sensor_name,
            lazy_loader=DGPFrameLazyLoader(
                unique_cache_key_prefix=unique_cache_key,
                dataset_path=self._dataset_path,
                custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
                scene_name=scene_name,
                sensor_name=sensor_name,
                calibration_key=sample.calibration_key,
                datum=scene_data.datum,
            ),
        )
        return sensor_frame

    def decode_ego_frame(self, scene_name: SceneName, frame_id: FrameId) -> EgoFrame:
        unique_cache_key = f"{self._dataset_path}-{scene_name}-{frame_id}-ego_frame"

        def _load_pose() -> EgoPose:
            sensor_name = next(iter(self.decode_available_sensor_names(scene_name=scene_name, frame_id=frame_id)))
            sensor_frame = self.decode_sensor_frame(scene_name=scene_name, frame_id=frame_id, sensor_name=sensor_name)
            vehicle_pose = sensor_frame.pose @ sensor_frame.extrinsic.inverse
            return EgoPose(quaternion=vehicle_pose.quaternion, translation=vehicle_pose.translation)

        return EgoFrame(unique_cache_key=unique_cache_key, pose_loader=_load_pose)

    @staticmethod
    def _scene_sample_to_date_time(sample: SceneSampleDTO) -> datetime:
        return iso8601.parse_date(sample.id.timestamp)
