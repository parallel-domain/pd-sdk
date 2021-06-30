import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union

from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp.constants import ANNOTATION_TYPE_MAP, DEFAULT_CLASS_MAP
from paralleldomain.decoding.dgp.dtos import DatasetDTO, DatasetMetaDTO, SceneDataDTO, SceneDTO, SceneSampleDTO
from paralleldomain.decoding.dgp.frame_lazy_loader import DGPFrameLazyLoader
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
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
        custom_map: Optional[ClassMap] = None,
        custom_id_map: Optional[ClassIdMap] = None,
        custom_reference_to_box_bottom: Optional[Transformation] = None,
    ):
        if custom_id_map is not None and custom_map is None:
            raise ValueError("A custom map has to be provided in order to match the custom id map!")

        if custom_id_map is not None and custom_map is not None:
            custom_map_ids = custom_map.class_ids
            if not all([target in custom_map_ids for target in custom_id_map.target_ids]):
                missing = set(custom_id_map.target_ids) - set(custom_map_ids)
                raise ValueError(
                    f"Not all target ids in the given custom id map are present in the custom map! Missing: {missing}"
                )

        self.custom_id_map = custom_id_map
        self.class_map = DEFAULT_CLASS_MAP if custom_map is None else custom_map
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
        scene_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_json_path.exists():
            files_with_prefix = [name.name for name in dataset_cloud_path.iterdir() if "scene_dataset" in name.name]
            if len(files_with_prefix) == 0:
                logger.error(
                    f"No scene_dataset.json or file starting with scene_dataset found under {dataset_cloud_path}!"
                )
            scene_json_path: AnyPath = dataset_cloud_path / files_with_prefix[-1]

        with scene_json_path.open(mode="r") as f:
            scene_dataset = json.load(f)

        meta_data = DatasetMetaDTO.from_dict(scene_dataset["metadata"])
        scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]
        return DatasetDTO(meta_data=meta_data, scene_names=scene_names)

    def decode_scene(self, scene_name: str) -> SceneDTO:
        scene_folder = self._dataset_path / scene_name
        potential_scene_files = [
            name.name for name in scene_folder.iterdir() if name.name.startswith("scene") and name.name.endswith("json")
        ]

        if len(potential_scene_files) == 0:
            logger.error(f"No sceneXXX.json found under {scene_folder}!")

        scene_file = scene_folder / potential_scene_files[0]
        with scene_file.open("r") as f:
            scene_data = json.load(f)
            scene_dto = SceneDTO.from_dict(scene_data)
            return scene_dto

    # ------------------------------------------------
    def get_unique_scene_id(self, scene_name: SceneName) -> str:
        return f"{self._dataset_path}-{scene_name}"

    def decode_scene_names(self) -> List[SceneName]:
        dto = self.decode_dataset()
        return [AnyPath(path).parent.name for path in dto.scene_names]

    def decode_dataset_meta_data(self) -> DatasetMeta:
        dto = self.decode_dataset()
        meta_dict = dto.meta_data.to_dict()
        anno_types = [ANNOTATION_TYPE_MAP[str(a)] for a in dto.meta_data.available_annotation_types]
        return DatasetMeta(name=dto.meta_data.name, available_annotation_types=anno_types, custom_attributes=meta_dict)

    def decode_scene_description(self, scene_name: SceneName) -> str:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return scene_dto.description

    def decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return {sample.id.index: self._scene_sample_to_date_time(sample=sample) for sample in scene_dto.samples}

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return list({datum.id.name for datum in scene_dto.data})

    def decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return list({datum.id.name for datum in scene_dto.data if datum.datum.image})

    def decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return list({datum.id.name for datum in scene_dto.data if datum.datum.point_cloud})

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
                class_map=self.class_map,
                custom_id_map=self.custom_id_map,
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
        try:
            return datetime.strptime(sample.id.timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            return datetime.strptime(sample.id.timestamp, "%Y-%m-%dT%H:%M:%SZ")
