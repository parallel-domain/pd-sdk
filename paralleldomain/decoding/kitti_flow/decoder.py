from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.kitti_flow.common import frame_id_to_timestamp
from paralleldomain.decoding.kitti_flow.frame_decoder import KITTIFlowFrameDecoder
from paralleldomain.decoding.kitti_flow.sensor_decoder import KITTIFlowCameraSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

IMAGE_FOLDER_NAME = "image_2"
OCC_OPTICAL_FLOW_FOLDER_NAME = "flow_occ"
NOC_OPTICAL_FLOW_FOLDER_NAME = "flow_noc"
AVAILABLE_ANNOTATION_IDENTIFIERS = [AnnotationIdentifier(annotation_type=AnnotationTypes.OpticalFlow)]


class KITTIFlowDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        split_name: str = "training",
        settings: Optional[DecoderSettings] = None,
        image_folder: Optional[str] = IMAGE_FOLDER_NAME,
        occ_optical_flow_folder: Optional[str] = OCC_OPTICAL_FLOW_FOLDER_NAME,
        noc_optical_flow_folder: Optional[str] = NOC_OPTICAL_FLOW_FOLDER_NAME,
        use_non_occluded: bool = False,
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
            image_folder=image_folder,
            occ_optical_flow_folder=occ_optical_flow_folder,
            noc_optical_flow_folder=noc_optical_flow_folder,
            use_non_occluded=use_non_occluded,
            camera_name="default",
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path) / split_name

        self.image_folder = image_folder
        self.occ_optical_flow_folder = occ_optical_flow_folder
        self.noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded
        self.camera_name = "default"
        dataset_name = "-".join(list([str(dataset_path), split_name]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return KITTIFlowSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            image_folder=self.image_folder,
            occ_optical_flow_folder=self.occ_optical_flow_folder,
            noc_optical_flow_folder=self.noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            camera_name=self.camera_name,
            scene_name=scene_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        scene_images_folder = self._dataset_path / self.image_folder
        # [:-7] removes _10.png or _11.png for first and second images in pairs.
        # We don't want to pull second images since they don't have a following image.
        scenes = {path.name[:-7] for path in scene_images_folder.iterdir()}
        return list(scenes)

    def _decode_scene_names(self) -> List[SceneName]:
        # May need to sort here depending on unordered scene logic
        return self._decode_unordered_scene_names()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=AVAILABLE_ANNOTATION_IDENTIFIERS,
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "kitti-flow"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class KITTIFlowSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
        image_folder: str,
        occ_optical_flow_folder: str,
        noc_optical_flow_folder: str,
        use_non_occluded: bool,
        camera_name: str,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings, scene_name=scene_name)
        self._image_folder = image_folder
        self._occ_optical_flow_folder = occ_optical_flow_folder
        self._noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded
        self._camera_name = camera_name

    def _decode_set_metadata(self) -> Dict[str, Any]:
        metadata_dict = dict(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            dataset_path=self._dataset_path,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            camera_name=self._camera_name,
        )
        return metadata_dict

    def _decode_set_description(self) -> str:
        return ""

    def _decode_frame_id_set(self) -> Set[FrameId]:
        frame_ids = {self.scene_name + "_10.png", self.scene_name + "_11.png"}
        return frame_ids

    def _decode_sensor_names(self) -> List[SensorName]:
        return [self._camera_name]

    def _decode_camera_names(self) -> List[SensorName]:
        return [self._camera_name]

    def _decode_lidar_names(self) -> List[SensorName]:
        return []

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return dict()

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[datetime]:
        return KITTIFlowCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            scene_decoder=self,
            is_unordered_scene=False,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[datetime]:
        raise ValueError("KITTI-flow does not support lidar data!")

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[datetime]:
        return KITTIFlowFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            frame_id=frame_id,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            camera_name=self._camera_name,
            scene_decoder=self,
            is_unordered_scene=False,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        fids = self._decode_frame_id_set()
        return {fid: frame_id_to_timestamp(frame_id=fid) for fid in fids}

    def _decode_radar_names(self) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[datetime]:
        raise ValueError("KITTI-flow does not support radar data!")

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return AVAILABLE_ANNOTATION_IDENTIFIERS
