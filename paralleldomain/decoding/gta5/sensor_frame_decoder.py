from typing import Dict, List, Optional

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image


class GTACameraSensorFrameDecoder(DirectoryCameraSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            dataset_path=dataset_path,
            settings=settings,
            folder_to_data_type=folder_to_data_type,
            metadata_folder=metadata_folder,
            class_map=class_map,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )

    def _decode_semantic_segmentation_2d(self) -> np.ndarray:
        annotation_path = (
            self._dataset_path
            / self._data_type_to_folder_name[AnnotationTypes.SemanticSegmentation2D]
            / f"{self.frame_id}.{self._img_file_extension}"
        )
        class_ids = read_image(path=annotation_path, convert_to_rgb=True, is_indexed=True).astype(int)
        return np.expand_dims(class_ids, axis=-1)
