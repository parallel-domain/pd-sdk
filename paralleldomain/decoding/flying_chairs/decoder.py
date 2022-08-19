from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.flying_chairs.common import frame_id_to_timestamp
from paralleldomain.decoding.flying_chairs.frame_decoder import FlyingChairsFrameDecoder
from paralleldomain.decoding.flying_chairs.sensor_decoder import FlyingChairsCameraSensorDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

FLYING_CHAIRS_DATASET_PATH = "s3://pd-internal-ml/flow/FlyingChairs"
IMAGE_FOLDER_NAME = "data"
OPTICAL_FLOW_FOLDER_NAME = "data"


class FlyingChairsDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath] = FLYING_CHAIRS_DATASET_PATH,
        split_name: str = "training",
        settings: Optional[DecoderSettings] = None,
        image_folder: Optional[str] = IMAGE_FOLDER_NAME,
        optical_flow_folder: Optional[str] = OPTICAL_FLOW_FOLDER_NAME,
        camera_name: Optional[str] = "default",
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
            image_folder=image_folder,
            optical_flow_folder=optical_flow_folder,
            camera_name=camera_name,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path) / split_name

        self.image_folder = image_folder
        self.optical_flow_folder = optical_flow_folder
        self.camera_name = camera_name
        dataset_name = "-".join(list([dataset_path, split_name]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return FlyingChairsSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            image_folder=self.image_folder,
            optical_flow_folder=self.optical_flow_folder,
            camera_name=self.camera_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        """
        Image filenames are of the form xxxxx_img1.ppm and xxxxx_img2.ppm.
        Flow filenames are of the form xxxxx_flow.flo.
        """
        # TODO Handle train/val split here
        scene_images_folder = self._dataset_path / self.image_folder
        scenes = {path.name[:5] for path in scene_images_folder.iterdir()}
        return list(scenes)

    def _decode_scene_names(self) -> List[SceneName]:
        return self._decode_unordered_scene_names()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "flying-chairs"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class FlyingChairsSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        settings: DecoderSettings,
        image_folder: str,
        optical_flow_folder: str,
        camera_name: str,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._image_folder = image_folder
        self._optical_flow_folder = optical_flow_folder
        self._camera_name = camera_name

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        metadata_dict = dict(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.OpticalFlow],
            dataset_path=self._dataset_path,
            image_folder=self._image_folder,
            optical_flow_folder=self._optical_flow_folder,
            camera_name=self._camera_name,
        )
        return metadata_dict

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        frame_ids = {scene_name + "_img1.ppm", scene_name + "_img2.ppm"}
        return frame_ids

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise ValueError("FlyingChairs decoder does not currently support lidar data!")

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {AnnotationTypes.OpticalFlow: None}

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[datetime]:
        return FlyingChairsCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            image_folder=self._image_folder,
            optical_flow_folder=self._optical_flow_folder,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        raise ValueError("Directory decoder does not support lidar data!")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return FlyingChairsFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            optical_flow_folder=self._optical_flow_folder,
            camera_name=self._camera_name,
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
