from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.flying_things.common import (
    CLEAN_IMAGE_FOLDER_1_NAME,
    FINAL_IMAGE_FOLDER_1_NAME,
    LEFT_SENSOR_NAME,
    OPTICAL_FLOW_FOLDER_NAME,
    RIGHT_SENSOR_NAME,
    SPLIT_NAME_TO_FOLDER_NAME,
    frame_id_to_timestamp,
    get_scene_folder,
)
from paralleldomain.decoding.flying_things.frame_decoder import FlyingThingsFrameDecoder
from paralleldomain.decoding.flying_things.sensor_decoder import FlyingThingsCameraSensorDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class FlyingThingsDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        split_name: str = "training",
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        sub_folders = list(self._dataset_path.iterdir())
        if "FlyingThings3D" in sub_folders:
            self._dataset_path = self._dataset_path / "FlyingThings3D"

        self.split_name = SPLIT_NAME_TO_FOLDER_NAME[split_name]
        dataset_name = "-".join(list([dataset_path, split_name]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return FlyingThingsSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            split_name=self.split_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self._decode_unordered_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        folder_path = self._dataset_path / OPTICAL_FLOW_FOLDER_NAME / self.split_name
        scenes_names = list(folder_path.iterdir())
        clean_scenes = [f"{CLEAN_IMAGE_FOLDER_1_NAME}/{n}" for n in scenes_names]
        final_scenes = [f"{FINAL_IMAGE_FOLDER_1_NAME}/{n}" for n in scenes_names]
        return clean_scenes + final_scenes

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "flying-things"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class FlyingThingsSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        settings: DecoderSettings,
        split_name: str,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._split_name = split_name

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        metadata_dict = dict(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            dataset_path=self._dataset_path,
            split_name=self._split_name,
            scene_name=scene_name,
        )
        return metadata_dict

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        folder_path = (
            get_scene_folder(dataset_path=self._dataset_path, scene_name=scene_name, split_name=self._split_name)
            / LEFT_SENSOR_NAME
        )
        frame_ids = {img.split(".png")[0] for img in folder_path.glob("*.png")}
        return frame_ids

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._decode_camera_names(scene_name=scene_name)

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise ValueError("FlyingThings decoder does not currently support lidar data!")

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return dict()

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[datetime]:
        return FlyingThingsCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            split_name=self._split_name,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        raise ValueError("Directory decoder does not support lidar data!")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return FlyingThingsFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            split_name=self._split_name,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        fids = self._decode_frame_id_set(scene_name=scene_name)
        return {fid: frame_id_to_timestamp(frame_id=fid) for fid in fids}

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[datetime]:
        raise ValueError("Loading from directory does not support radar data!")
