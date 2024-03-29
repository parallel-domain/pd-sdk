import logging
from typing import Dict, List, Optional, Union

from paralleldomain.decoding.decoder import SceneDecoder
from paralleldomain.decoding.directory.decoder import DirectoryDatasetDecoder, DirectorySceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.kitti.frame_decoder import KittiFrameDecoder
from paralleldomain.decoding.kitti.sensor_decoder import KittiLidarSensorDecoder
from paralleldomain.decoding.sensor_decoder import LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)

FOLDER_TO_DATA_TYPE = {
    "velodyne": PointCloud,
    "label_2": AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D),
}

KITTI_CLASS_MAP = [
    ClassDetail(name="Car", id=0, instanced=True),
    ClassDetail(name="Van", id=2, instanced=True),
    ClassDetail(name="Truck", id=3, instanced=True),
    ClassDetail(name="Pedestrian", id=4, instanced=True),
    ClassDetail(name="Person (sitting)", id=5, instanced=True),
    ClassDetail(name="Cyclist", id=7, instanced=True),
    ClassDetail(name="Tram", id=8, instanced=True),
    ClassDetail(name="Misc", id=9, instanced=True),
    ClassDetail(name="DontCare", id=10, instanced=True),
]


class KittiDatasetDecoder(DirectoryDatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        class_map: Optional[List[ClassDetail]] = None,
        folder_to_data_type: Optional[Dict[str, SensorDataCopyTypes]] = None,
        point_cloud_dim: Optional[int] = 4,
        **kwargs,
    ):
        class_map = KITTI_CLASS_MAP if class_map is None else class_map
        folder_to_data_type = FOLDER_TO_DATA_TYPE if folder_to_data_type is None else folder_to_data_type
        self.point_cloud_dim = point_cloud_dim
        super().__init__(
            dataset_path=dataset_path, class_map=class_map, folder_to_data_type=folder_to_data_type, **kwargs
        )

    @staticmethod
    def get_format() -> str:
        return "kitti"

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return ["default_scene"]

    def _decode_scene_names(self) -> List[SceneName]:
        return ["default_scene"]

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=[AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D)],
            custom_attributes=dict(),
        )

    def create_scene_decoder(self, scene_name: SceneName) -> SceneDecoder:
        return KittiDirectorySceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            class_map=self.class_map,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self.metadata_folder,
            sensor_name=self.sensor_name,
            point_cloud_dim=self.point_cloud_dim,
            scene_name=scene_name,
        )


class KittiDirectorySceneDecoder(DirectorySceneDecoder):
    def __init__(
        self,
        scene_name: SceneName,
        point_cloud_dim: int,
        **kwargs,
    ):
        self.point_cloud_dim = point_cloud_dim
        super().__init__(scene_name=scene_name, **kwargs)

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[None]:
        return KittiLidarSensorDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            point_cloud_dim=self.point_cloud_dim,
            scene_decoder=self,
            is_unordered_scene=True,
        )

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[None]:
        return KittiFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            sensor_name=self._sensor_name,
            class_map=self._class_map,
            point_cloud_dim=self.point_cloud_dim,
            scene_decoder=self,
            is_unordered_scene=True,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, None]:
        return {fid: int(fid) for fid in self.get_frame_ids()}
