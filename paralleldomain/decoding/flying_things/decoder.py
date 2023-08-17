from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.flying_things.common import (
    CLEAN_IMAGE_FOLDER_1_NAME,
    FINAL_IMAGE_FOLDER_1_NAME,
    LEFT_SENSOR_NAME,
    OPTICAL_FLOW_FOLDER_NAME,
    RIGHT_SENSOR_NAME,
    SPLIT_NAME_TO_FOLDER_NAME,
    decode_frame_id_set,
    frame_id_to_timestamp,
)
from paralleldomain.decoding.flying_things.frame_decoder import FlyingThingsFrameDecoder
from paralleldomain.decoding.flying_things.sensor_decoder import FlyingThingsCameraSensorDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

AVAILABLE_ANNOTATION_IDENTIFIERS = [
    AnnotationIdentifier(annotation_type=AnnotationTypes.OpticalFlow),
    AnnotationIdentifier(annotation_type=AnnotationTypes.BackwardOpticalFlow),
]


class FlyingThingsDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        split_name: str = "training",
        settings: Optional[DecoderSettings] = None,
        is_full_dataset_format: bool = False,
        is_driving_subset: bool = False,
        train_split_file: Optional[AnyPath] = None,
        val_split_file: Optional[AnyPath] = None,
        **kwargs,
    ):
        """
        Format Definition see here:
        https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#information
        This decoder supports both formats of Flying Things that are available. For one the Full Dataset format with
        the structure:
        dataset_path
            frames_cleanpass
                TEST
                    A
                        scene_folder_xy
                            image_clean
                                left: X images
                                right: X images
                        .
                        .
                    B
                    C
                TRAIN
                    A
                        scene_folder_xy
                            image_clean
                                left: X images
                                right: X images
                        .
                        .
                    B
                    C
            frames_finalpass
                TEST
                    A
                        scene_folder_xy
                            image_final
                                left: X images
                                right: X images
                        .
                        .
                    B
                    C
                TRAIN
                    .
                    .
            optical_flow
                TEST
                    A
                        scene_folder_xy
                            into_future
                                left: X images
                                right: X images
                            into_past
                                left: X images
                                right: X images
                        .
                        .
                    B
                    C
                TRAIN
                    .
                    .
        where each folder under TEST or TRAIN contain a scene. And the Format of the DispNet/FlowNet2.0 dataset subsets,
        which has no per scene folders but a split file. This Format would have the following folder structure:
        dataset_path
            frames_cleanpass
                train
                    image_clean
                        left: 21,818 imgs
                        right: 21,818 imgs
                val
                    image_clean
                        left: 4,248 imgs
                        right: 4,248 imgs
            frames_finalpass
                train
                    image_final
                        left: 21,818 imgs
                        right: 21,818 imgs
                val
                    image_final
                        left: 4,248 imgs
                        right: 4,248 imgs
            optical_flow
                train
                    left
                        into_past: 19,642 flo
                        into_future: 19,642 flo
                    right
                        into_past: 19,642 flo
                        into_future: 19,642 flo
                val
                    left
                        into_past: 3,824 flo
                        into_future: 3,824 flo
                    right
                        into_past: 3,824 flo
                        into_future: 3,824 flo

        Args:
        dataset_path: The path to the root folder of the dataset. See Above description.
        split_name: One of `training`, `train`, `validation`, `val`, `testing`, `test`. Depending on the Format -
            Full vs Subset, either test or val is available. E.g. the full dataset only has train and test splits, while
            the DispNet/FlowNet2.0 dataset subsets has train and val.
        settings: Optional settings for the decoder. For Details see DecoderSettings.
        is_full_dataset_format: A boolean flag to indicate if the dataset_path points to the Full dataset format or
            the DispNet/FlowNet2.0 dataset subsets format.
        is_driving_subset: Since the Driving Subset of the official dataset has a different focal length from the other
            subsets, this flag indicates the decoder to use the changed camera intrinsics (fx/fy = 450 instead of 1050).
        train_split_file: Optional path to a file that contains the scene splits in case of the
            DispNet/FlowNet2.0 dataset subsets format train split. This is only needed if is_full_dataset_format
            is False. If no path is passed we assume that the file is located under
            dataset_path/sequence-lengths-train.txt
            This file can be found in the Sequence lengths column of the DispNet/FlowNet2.0 dataset subsets
            documentation of the official website.
        val_split_file: Optional path to a file that contains the scene splits in case of the
            DispNet/FlowNet2.0 dataset subsets format val split.  This is only needed if is_full_dataset_format
            is False. If no path is passed we assume that the file is located under
            dataset_path/sequence-lengths-train.txt
            This file can be found in the Sequence lengths column of the DispNet/FlowNet2.0 dataset subsets
            documentation of the official website.
        """

        self.is_driving_subset = is_driving_subset
        self.val_split_file = val_split_file
        self.train_split_file = train_split_file
        self.is_full_dataset_format = is_full_dataset_format
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        if self.val_split_file is None and not is_full_dataset_format:
            self.val_split_file = self._dataset_path / "sequence-lengths-val.txt"

        if self.train_split_file is None and not is_full_dataset_format:
            self.train_split_file = self._dataset_path / "sequence-lengths-train.txt"

        self.split_name: str = SPLIT_NAME_TO_FOLDER_NAME[split_name]
        if not self.is_full_dataset_format:
            self.split_name = self.split_name.lower()

        self.split_list: List[int] = list()

        if self.split_name in ["val", "train"]:
            split_file: AnyPath = {
                "val": self.val_split_file,
                "train": self.train_split_file,
            }[self.split_name]
            with split_file.open("r") as f:
                self.split_list = [int(line) for line in f.readlines()]

        dataset_name = "-".join(list([str(dataset_path), split_name]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return FlyingThingsSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            split_name=self.split_name,
            is_full_dataset_format=self.is_full_dataset_format,
            split_list=self.split_list,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        has_clean = (self._dataset_path / CLEAN_IMAGE_FOLDER_1_NAME).exists()
        has_final = (self._dataset_path / FINAL_IMAGE_FOLDER_1_NAME).exists()
        scene_names = list()
        if self.is_full_dataset_format:
            names = list()
            for sub_split in ["A", "B", "C"]:
                folder_path = self._dataset_path / OPTICAL_FLOW_FOLDER_NAME / self.split_name / sub_split
                names += [f"{sub_split}/{n}" for n in folder_path.iterdir()]

            clean_scenes = [f"{CLEAN_IMAGE_FOLDER_1_NAME}/{n}" for n in names]
            final_scenes = [f"{FINAL_IMAGE_FOLDER_1_NAME}/{n}" for n in names]
        else:
            clean_scenes = [f"{CLEAN_IMAGE_FOLDER_1_NAME}/{i}" for i in range(len(self.split_list))]
            final_scenes = [f"{FINAL_IMAGE_FOLDER_1_NAME}/{i}" for i in range(len(self.split_list))]

        if has_clean:
            scene_names += clean_scenes
        if has_final:
            scene_names += final_scenes
        return scene_names

    def _decode_scene_names(self) -> List[SceneName]:
        return self._decode_unordered_scene_names()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=AVAILABLE_ANNOTATION_IDENTIFIERS,
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
        split_list: List[int],
        is_full_dataset_format: bool = False,
        is_driving_subset: bool = False,
    ):
        self._is_full_dataset_format = is_full_dataset_format
        self._is_driving_subset = is_driving_subset
        self._split_list = split_list
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

    def _decode_available_annotation_identifiers(self, scene_name: SceneName) -> List[AnnotationIdentifier]:
        return AVAILABLE_ANNOTATION_IDENTIFIERS

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        return decode_frame_id_set(
            scene_name=scene_name,
            split_name=self._split_name,
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            dataset_path=self._dataset_path,
            sensor_name=LEFT_SENSOR_NAME,
        )

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._decode_camera_names(scene_name=scene_name)

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise ValueError("FlyingThings decoder does not currently support lidar data!")

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationIdentifier, ClassMap]:
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
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            is_driving_subset=self._is_driving_subset,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        raise ValueError("FlyingThings does not support lidar data!")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return FlyingThingsFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            split_name=self._split_name,
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            is_driving_subset=self._is_driving_subset,
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
        raise ValueError("FlyingThings does not support radar data!")
