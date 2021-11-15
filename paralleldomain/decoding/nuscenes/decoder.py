from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from iso8601 import iso8601

import paralleldomain.decoding.nuscenes.splits as nu_splits
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.nuscenes.common import NUSCENES_CLASSES, NuScenesDataAccessMixin, load_table
from paralleldomain.decoding.nuscenes.frame_decoder import NuScenesFrameDecoder
from paralleldomain.decoding.nuscenes.sensor_decoder import NuScenesCameraSensorDecoder, NuScenesLidarSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

SPLIT_NAME_TO_NU_SPLIT = {
    "train": "v1.0-trainval",
    "val": "v1.0-trainval",
    "test": "v1.0-test",
    "mini-train": "v1.0-mini",
    "mini-val": "v1.0-mini",
}


class NuScenesDatasetDecoder(DatasetDecoder, NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: Optional[DecoderSettings] = None,
        nu_split_name: Optional[str] = None,
        split_name: Optional[str] = None,
        **kwargs,
    ):
        """Decodes a NuScenes dataset

        Args:
            dataset_path: AnyPath to the root folder of a NuScenes dataset.
            nu_split_name: Split to use within this dataset. By default the matching split for split_name will
            be picket from SPLIT_NAME_TO_NU_SPLIT. Other Options are ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
            split_name: The scenes split within the split. For example in "v1.0-trainval" are train and
             validation samples. To access the validation samples pass "val". Options are
             ["mini-train", "mini-val", "test", "val", "train"]. Defaults to "mini_train"
        """
        self.settings = settings
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        if split_name is None:
            split_name = "mini_train"
        if nu_split_name is None:
            nu_split_name = SPLIT_NAME_TO_NU_SPLIT[split_name]
        self.split_name = split_name
        self.nu_split_name = nu_split_name
        self.split_scene_names = getattr(nu_splits, split_name)
        dataset_name = f"NuScenes-{nu_split_name}-{split_name}"
        DatasetDecoder.__init__(self=self, dataset_name=dataset_name, settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=nu_split_name, dataset_path=self._dataset_path
        )

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return NuScenesSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            split_name=self.split_name,
            settings=self.settings,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self.get_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        return [nu_s["name"] for nu_s in self.nu_scene if nu_s["name"] in self.split_scene_names]

    # Update this function when lidar_semseg is added.
    def _decode_dataset_metadata(self) -> DatasetMeta:
        available_annotation_types = list()
        if self.split_name != "v1.0-test" or (len(self.nu_sample_annotation) > 0):
            available_annotation_types = [
                # AnnotationTypes.SemanticSegmentation3D,
                AnnotationTypes.BoundingBoxes3D,
            ]

        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=available_annotation_types,
            custom_attributes=dict(split_name=self.split_name),
        )


class NuScenesSceneDecoder(SceneDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self, dataset_path: Union[str, AnyPath], dataset_name: str, split_name: str, settings: DecoderSettings
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        SceneDecoder.__init__(self=self, dataset_name=str(dataset_path), settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        # Because there are multiple nuScenes scenes in a log entry, this combines the log and scene metadata.
        scene_metadata = self.nu_scene_by_scene_name[scene_name]
        log_metadata = self.nu_logs_by_log_token[scene_metadata["log_token"]]
        return {**log_metadata, **scene_metadata}

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return self.nu_scene_by_scene_name[scene_name]["description"]

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        scene_token = self.nu_scene_name_to_scene_token[scene_name]
        return {sample["token"] for sample in self.nu_samples[scene_token]}

    def _decode_sensor_names_by_modality(
        self, scene_name: SceneName, modality: List[str] = ["camera", "lidar"]
    ) -> List[SensorName]:
        scene_token = self.nu_scene_name_to_scene_token[scene_name]
        samples = self.nu_samples[scene_token]
        sample_tokens = [sample["token"] for sample in samples]
        sensor_names = set()

        data_dict = self.nu_samples_data
        for sample_token in sample_tokens:
            data_list = data_dict[sample_token]
            for data in data_list:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["modality"] in modality:
                    sensor_names.add(sensor["channel"])
        return list(sensor_names)

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._decode_sensor_names_by_modality(scene_name=scene_name, modality=["camera", "lidar"])

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._decode_sensor_names_by_modality(scene_name=scene_name, modality=["camera"])

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._decode_sensor_names_by_modality(scene_name=scene_name, modality=["lidar"])

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {
            # AnnotationTypes.SemanticSegmentation3D: ClassMap(classes=self.nu_class_infos),
            AnnotationTypes.BoundingBoxes3D: ClassMap(classes=self.nu_class_infos),
        }

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[datetime]:
        return NuScenesCameraSensorDecoder(
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=self.split_name,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        return NuScenesLidarSensorDecoder(
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=self.split_name,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return NuScenesFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=self.split_name,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        scene_token = self.nu_scene_name_to_scene_token[scene_name]
        samples = self.nu_samples[scene_token]
        return {s["token"]: datetime.fromtimestamp(int(s["timestamp"]) / 1000000) for s in samples}
