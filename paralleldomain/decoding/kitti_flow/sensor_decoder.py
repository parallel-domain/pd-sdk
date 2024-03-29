from datetime import datetime
from functools import lru_cache
from typing import Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.kitti_flow.sensor_frame_decoder import KITTIFlowCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class KITTIFlowCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        occ_optical_flow_folder: str,
        noc_optical_flow_folder: str,
        use_non_occluded: bool,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._occ_optical_flow_folder = occ_optical_flow_folder
        self._noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded

    def _decode_frame_id_set(self) -> Set[FrameId]:
        frame_ids = {self.scene_name + "_10.png", self.scene_name + "_11.png"}
        return frame_ids

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return KITTIFlowCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self._dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
