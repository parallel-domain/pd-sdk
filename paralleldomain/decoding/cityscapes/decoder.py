from typing import Any, Dict, List, Optional, Union

from paralleldomain.decoding.cityscapes.scene_decoder import CityscapesSceneDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath

_AVAILABLE_ANNOTATION_TYPES = [AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D]
_AVAILABLE_ANNOTATION_IDENTIFIERS = [AnnotationIdentifier(annotation_type=t) for t in _AVAILABLE_ANNOTATION_TYPES]
IMAGE_FOLDER_NAME = "leftImg8bit"


class CityscapesDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        splits: Optional[List[str]] = None,
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        self._init_kwargs = dict(dataset_path=dataset_path, settings=settings, splits=splits)
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        if splits is None:
            splits = ["test", "train", "val"]
        self.splits = splits
        dataset_name = "-".join(list(["cityscapes"] + splits))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> CityscapesSceneDecoder:
        return CityscapesSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            scene_name=scene_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        scene_names = list()
        for split_name in self.splits:
            split_scenes_folder = self._dataset_path / IMAGE_FOLDER_NAME / split_name
            for folder_path in split_scenes_folder.iterdir():
                scene_name = f"{split_name}-{folder_path.name}"
                scene_names.append(scene_name)
        return scene_names

    def _decode_scene_names(self) -> List[SceneName]:
        return list()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=_AVAILABLE_ANNOTATION_IDENTIFIERS,
            custom_attributes=dict(splits=self.splits),
        )

    @staticmethod
    def get_format() -> str:
        return "cityscapes"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs
