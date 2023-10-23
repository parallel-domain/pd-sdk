from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_INDEX_TO_CAMERA_NAME,
    WAYMO_USE_ALL_LIDAR_NAME,
    decode_class_maps,
    get_cached_pre_calculated_scene_to_frame_info,
    get_record_iterator,
)
from paralleldomain.decoding.waymo_open_dataset.frame_decoder import WaymoOpenDatasetFrameDecoder
from paralleldomain.decoding.waymo_open_dataset.sensor_decoder import (
    WaymoOpenDatasetCameraSensorDecoder,
    WaymoOpenDatasetLidarSensorDecoder,
)
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

IMAGE_FOLDER_NAME = "image"
SEMANTIC_SEGMENTATION_FOLDER_NAME = "semantic_segmentation"
METADATA_FOLDER_NAME = "metadata"
AVAILABLE_ANNOTATION_TYPES = [
    AnnotationTypes.SemanticSegmentation2D,
    AnnotationTypes.InstanceSegmentation2D,
    AnnotationTypes.BoundingBoxes2D,
    AnnotationTypes.BoundingBoxes3D,
]
AVAILABLE_ANNOTATION_IDENTIFIERS = [AnnotationIdentifier(annotation_type=t) for t in AVAILABLE_ANNOTATION_TYPES]


class WaymoOpenDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        split_name: str,
        settings: Optional[DecoderSettings] = None,
        use_precalculated_maps: bool = True,
        include_second_returns: bool = True,
        index_folder: Optional[AnyPath] = None,
        **kwargs,
    ):
        """
        A decoder for the waymo open dataset format version 1.4.0.
        Args:
            dataset_path: The root folder of the dataset that contains a sub folder named training and validation.
            split_name: the sub folder to pick from the dataset_path. Either training or validation.
                If you have differently named sub splits you can also pass them here. The default format just has
                training and validation.
            settings: The decoder settings you can use to turn on caching of images annotations etc and to pass
                custom wrappers to model classes.
            use_precalculated_maps: If you want to use precalcualted indices that speed up the lookup of frame ids
                as well as the available annotations on each frame. By default, we would need to download an entire
                scene record and parse through it to know which annotations exist.
            include_second_returns: If the return point cloud should include second return of the lidar
            index_folder: The folder that contains the pre-calculated maps. If this is None we will check if the
                dataset_path/precomputed_indices exists and use that. If that does not exist we will
                not support using precalculated maps. FOr each split we expect 3 files to be present with the name
                <split_name>_scene_to_frame_info.json, <split_name>_sensor_frame_to_has_bounding_box_2d.json and
                <split_name>_sensor_frame_to_has_segmentation.json.
                <split_name>_scene_to_frame_info.json contains a map from scene name to a list of
                dictionaries with the keys
                timestamp_micros: int and frame_id: str.
                <split_name>_sensor_frame_to_has_segmentation.json
                and <split_name>_sensor_frame_to_has_bounding_box_2d.json contain a map from
                <scene_name>-<frame_id>-<sensor_name> to True if that sensor frame has that annotation.
        """
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
            **kwargs,
        )
        dataset_path = AnyPath(dataset_path)
        if index_folder is None:
            potential_index_folder = dataset_path / "precomputed_indices"
            if potential_index_folder.exists():
                index_folder = potential_index_folder

        self._dataset_path: AnyPath = dataset_path / split_name
        self.index_folder = index_folder
        self.split_name = split_name
        self.include_second_returns = include_second_returns
        self.use_precalculated_maps = use_precalculated_maps
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

        dataset_name = f"Waymo Open Dataset - {split_name}"
        super().__init__(dataset_name=dataset_name, settings=settings, **kwargs)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return WaymoOpenDatasetSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            settings=self.settings,
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
            include_second_returns=self.include_second_returns,
            index_folder=self.index_folder,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self.get_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        if self.use_precalculated_maps and self.index_folder is not None:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                index_folder=self.index_folder,
            )
            return sorted(list(id_map.keys()))

        return sorted([f.name for f in self._dataset_path.iterdir()])
        # return []

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=AVAILABLE_ANNOTATION_IDENTIFIERS,
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "waymo_open_dataset"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class WaymoOpenDatasetSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        include_second_returns: bool,
        index_folder: Optional[AnyPath],
    ):
        self.split_name = split_name
        self.include_second_returns = include_second_returns
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self.use_precalculated_maps = use_precalculated_maps
        self.index_folder = index_folder
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

        super().__init__(dataset_name=dataset_name, settings=settings, scene_name=scene_name)

    def _decode_set_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self) -> str:
        return ""

    def _decode_frame_id_set(self) -> Set[FrameId]:
        if self.use_precalculated_maps and self.index_folder is not None:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                index_folder=self.index_folder,
            )
            if self.scene_name in id_map:
                return {elem["frame_id"] for elem in id_map[self.scene_name]}
        record = self._dataset_path / self.scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        return set(frame_ids)

    def _decode_sensor_names(self) -> List[SensorName]:
        cam_names = self.get_camera_names()
        lidar_names = self.get_lidar_names()
        return cam_names + lidar_names

    def _decode_camera_names(self) -> List[SensorName]:
        return list(WAYMO_INDEX_TO_CAMERA_NAME.values())

    def _decode_lidar_names(self) -> List[SensorName]:
        return [WAYMO_USE_ALL_LIDAR_NAME]

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return decode_class_maps()

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[datetime]:
        return WaymoOpenDatasetCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            sensor_name=sensor_name,
            scene_name=self.scene_name,
            settings=self.settings,
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
            scene_decoder=self,
            is_unordered_scene=False,
            index_folder=self.index_folder,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[datetime]:
        return WaymoOpenDatasetLidarSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            settings=self.settings,
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
            include_second_returns=self.include_second_returns,
            scene_decoder=self,
            is_unordered_scene=False,
            index_folder=self.index_folder,
        )

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[datetime]:
        return WaymoOpenDatasetFrameDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            frame_id=frame_id,
            scene_name=self.scene_name,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            include_second_returns=self.include_second_returns,
            scene_decoder=self,
            is_unordered_scene=False,
            index_folder=self.index_folder,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        frame_id_to_date_time_map = dict()
        if self.use_precalculated_maps and self.index_folder is not None:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                index_folder=self.index_folder,
            )
            for elem in id_map[self.scene_name]:
                frame_id_to_date_time_map[elem["frame_id"]] = datetime.fromtimestamp(elem["timestamp_micros"] / 1000000)
        else:
            record = self._dataset_path / self.scene_name
            for record, frame_id in get_record_iterator(record_path=record, read_frame=True):
                frame_id_to_date_time_map[frame_id] = datetime.fromtimestamp(record.timestamp_micros / 1000000)
        return frame_id_to_date_time_map

    def _decode_radar_names(self) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[TDateTime]:
        raise ValueError("This dataset has no radar data!")

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return AVAILABLE_ANNOTATION_IDENTIFIERS
