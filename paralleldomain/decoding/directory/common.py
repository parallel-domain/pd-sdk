from functools import lru_cache
from typing import Dict, List, Optional

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath


def decode_class_maps(
    class_map: List[ClassDetail], annotation_types: List[AnnotationIdentifier]
) -> Dict[AnnotationIdentifier, ClassMap]:
    return {annotation_type: ClassMap(classes=class_map) for annotation_type in annotation_types}


@lru_cache(maxsize=10)
def _cached_dataset_contains_scene(dataset_path: AnyPath, scene_name: SceneName) -> bool:
    return any([item.name == scene_name for item in dataset_path.iterdir()])


def resolve_scene_folder(dataset_path: AnyPath, scene_name: Optional[SceneName]) -> AnyPath:
    if scene_name is None:
        return dataset_path
    elif _cached_dataset_contains_scene(dataset_path=dataset_path, scene_name=scene_name):
        return dataset_path / scene_name
    else:
        return dataset_path
