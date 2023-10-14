from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.nuimages.common import NuImagesDataAccessMixin
from paralleldomain.decoding.nuimages.frame_decoder import NuImagesFrameDecoder
from paralleldomain.decoding.nuimages.sensor_decoder import NuImagesCameraSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuImagesDatasetDecoder(DatasetDecoder, NuImagesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: Optional[DecoderSettings] = None,
        split_name: Optional[str] = None,
        **kwargs,
    ):
        """Decodes a NuImages dataset

        Args:
            dataset_path: AnyPath to the root folder of a NuImages dataset.
            split: Split to use within this dataset. Defaults to v1.0-train.
            Options are [v1.0-mini, v1.0-test, v1.0-train, v1.0-val].
        """
        self._init_kwargs = dict(dataset_path=dataset_path, settings=settings, split_name=split_name)
        self.settings = settings
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        if split_name is None:
            split_name = "v1.0-train"
        self.split_name = split_name
        dataset_name = f"NuImages-{split_name}"
        DatasetDecoder.__init__(self=self, dataset_name=dataset_name, settings=settings)
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return NuImagesSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            split_name=self.split_name,
            settings=self.settings,
            scene_name=scene_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self.get_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        scene_names = list()
        for log in self.nu_logs:
            scene_names.append(log["token"])
        return scene_names

    def _decode_dataset_metadata(self) -> DatasetMeta:
        available_annotation_types = list()
        if self.split_name != "v1.0-test" or (len(self.nu_surface_ann) > 0 and len(self.nu_object_ann) > 0):
            available_annotation_types = [
                AnnotationTypes.SemanticSegmentation2D,
                AnnotationTypes.InstanceSegmentation2D,
                AnnotationTypes.BoundingBoxes2D,
            ]
        available_annotation_identifiers = [AnnotationIdentifier(annotation_type=t) for t in available_annotation_types]

        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=available_annotation_identifiers,
            custom_attributes=dict(split_name=self.split_name),
        )

    @staticmethod
    def get_format() -> str:
        return "nuimages"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class NuImagesSceneDecoder(SceneDecoder[datetime], NuImagesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        SceneDecoder.__init__(self=self, dataset_name=str(dataset_path), settings=settings, scene_name=scene_name)
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_set_metadata(self) -> Dict[str, Any]:
        log = self.nu_logs_by_log_token[self.scene_name]
        return log

    def _decode_set_description(self) -> str:
        return ""

    def _decode_frame_id_set(self) -> Set[FrameId]:
        sample_data_ids = set()
        for sample in self.nu_samples[self.scene_name]:
            sample_data_ids.add(sample["key_camera_token"])
        nu_samples_data = self.nu_samples_data
        return {str(nu_samples_data[sample_id]["timestamp"]) for sample_id in sample_data_ids}

    def _decode_sensor_names(self) -> List[SensorName]:
        return self.get_camera_names()

    def _decode_camera_names(self) -> List[SensorName]:
        samples = self.nu_samples[self.scene_name]
        key_camera_tokens = [sample["key_camera_token"] for sample in samples]
        camera_names = set()
        data_dict = self.nu_samples_data
        for key_camera_token in key_camera_tokens:
            data = data_dict[key_camera_token]
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["modality"] == "camera":
                camera_names.add(sensor["channel"])
        return list(camera_names)

    def _decode_lidar_names(self) -> List[SensorName]:
        return list()

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self.nu_class_maps

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[datetime]:
        return NuImagesCameraSensorDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            sensor_name=sensor_name,
            split_name=self.split_name,
            scene_name=self.scene_name,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=False,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[datetime]:
        raise ValueError("NuImages does not contain lidar data!")

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[datetime]:
        return NuImagesFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            frame_id=frame_id,
            split_name=self.split_name,
            scene_name=self.scene_name,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=False,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        return {fid: datetime.fromtimestamp(int(fid) / 1000000) for fid in self.get_frame_ids()}

    def _decode_radar_names(self) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[None]:
        raise ValueError("NuImages does not contain radar data!")

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        if self.split_name == "v1.0-test":
            return []
        samples = self.nu_sample_data_tokens_to_available_anno_types
        has_surface = any(has_surface for has_surface, _ in samples.values())
        has_obj = any(has_obj for _, has_obj in samples.values())
        anno_identifiers = list()
        if has_surface:
            anno_identifiers.append(AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D))
        if has_obj:
            anno_identifiers.append(AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D))
            anno_identifiers.append(AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D))
        return anno_identifiers
