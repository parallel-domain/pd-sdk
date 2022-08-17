from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.kitti.frame_decoder import KITTIFrameDecoder
from paralleldomain.decoding.kitti.sensor_decoder import KITTICameraSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

KITTI_DATASET_PATH = "s3://pd-internal-ml/flow/KITTI2015"
IMAGE_FOLDER_NAME = "image_2"
OCC_OPTICAL_FLOW_FOLDER_NAME = "flow_occ"
NOC_OPTICAL_FLOW_FOLDER_NAME = "flow_noc"


class KITTIDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath] = KITTI_DATASET_PATH,
        split_name: str = "training",
        settings: Optional[DecoderSettings] = None,
        image_folder: Optional[str] = IMAGE_FOLDER_NAME,
        occ_optical_flow_folder: Optional[str] = OCC_OPTICAL_FLOW_FOLDER_NAME,
        noc_optical_flow_folder: Optional[str] = NOC_OPTICAL_FLOW_FOLDER_NAME,
        use_non_occluded: bool = False,
        camera_name: Optional[str] = "default",
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
            camera_name=camera_name,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path) / split_name

        self.image_folder = image_folder
        self.occ_optical_flow_folder = occ_optical_flow_folder
        self.noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded
        self.camera_name = camera_name
        dataset_name = "-".join(list([dataset_path, split_name]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return KITTISceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            image_folder=self.image_folder,
            occ_optical_flow_folder=self.occ_optical_flow_folder,
            noc_optical_flow_folder=self.noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            camera_name=self.camera_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [self.image_folder]

    def _decode_scene_names(self) -> List[SceneName]:
        return list()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "kitti"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class KITTISceneDecoder(SceneDecoder[None]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        settings: DecoderSettings,
        image_folder: str,
        occ_optical_flow_folder: str,
        noc_optical_flow_folder: str,
        use_non_occluded: bool,
        camera_name: str,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._image_folder = image_folder
        self._occ_optical_flow_folder = occ_optical_flow_folder
        self._noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded
        self._camera_name = camera_name

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
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

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        scene_images_folder = self._dataset_path / self._image_folder
        # [:-7] removes _10.png or _11.png for first and second images in pairs.
        # We don't want to pull second images since they don't have a following image.
        path_set = {path.name[:-7] for path in scene_images_folder.iterdir()}
        return {path + "_10.png" for path in path_set}

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise ValueError("KITTI decoder does not currently support lidar data!")

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return dict()

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[None]:
        return KITTICameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[None]:
        raise ValueError("Directory decoder does not support lidar data!")

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder[None]:
        return KITTIFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            camera_name=self._camera_name,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, None]:
        return {fid: None for fid in self.get_frame_ids(scene_name=scene_name)}

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")
