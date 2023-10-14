import logging
from typing import Any, Dict, List, Optional, Union

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.common import _DatasetDecoderMixin
from paralleldomain.decoding.dgp.scene_decoder import DGPSceneDecoder
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


class DGPDatasetDecoder(_DatasetDecoderMixin, DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        custom_reference_to_box_bottom: Optional[Transformation] = None,
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            settings=settings,
            custom_reference_to_box_bottom=custom_reference_to_box_bottom,
        )
        _DatasetDecoderMixin.__init__(self, dataset_path=dataset_path)
        DatasetDecoder.__init__(self, dataset_name=str(dataset_path), settings=settings)
        self.custom_reference_to_box_bottom = (
            Transformation() if custom_reference_to_box_bottom is None else custom_reference_to_box_bottom
        )

        self._dataset_path: AnyPath = AnyPath(dataset_path)

    def create_scene_decoder(self, scene_name: SceneName) -> DGPSceneDecoder:
        return DGPSceneDecoder(
            dataset_path=self._dataset_path,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            scene_name=scene_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [p.parent.name for p in self._decode_scene_paths()]

    def _decode_dataset_metadata(self) -> DatasetMeta:
        dto = self._decode_dataset_dto()
        meta_dict = dto.metadata.to_dict()
        anno_types = [ANNOTATION_TYPE_MAP[str(a)] for a in dto.metadata.available_annotation_types]
        anno_identifiers = [AnnotationIdentifier(annotation_type=t) for t in anno_types]
        return DatasetMeta(
            name=dto.metadata.name, available_annotation_identifiers=anno_identifiers, custom_attributes=meta_dict
        )

    @staticmethod
    def get_format() -> str:
        return "dgp"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs
