from typing import List, Optional

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.decoding.gta5.common import IMAGE_FOLDER_NAME, SEMANTIC_SEGMENTATION_FOLDER_NAME
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.type_aliases import FrameId, SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image


class GTACameraSensorFrameDecoder(DirectoryCameraSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=dataset_path,
            settings=settings,
            image_folder=IMAGE_FOLDER_NAME,
            semantic_segmentation_folder=SEMANTIC_SEGMENTATION_FOLDER_NAME,
            metadata_folder=metadata_folder,
            class_map=class_map,
        )

    def _decode_semantic_segmentation_2d(
        self, scene_name: str, frame_id: FrameId, annotation_identifier: str
    ) -> np.ndarray:
        annotation_path = self._dataset_path / self._semantic_segmentation_folder / f"{frame_id}"
        class_ids = read_image(path=annotation_path, convert_to_rgb=True, is_indexed=True).astype(int)
        return np.expand_dims(class_ids, axis=-1)
