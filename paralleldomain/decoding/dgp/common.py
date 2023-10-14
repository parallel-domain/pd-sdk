from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.common.dgp.v0.dtos import DatasetDTO, OntologyFileDTO
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json


# Caching on module level so that not _DatasetDecoderMixin instance is calling it.
# This gives significant speedups, unless you do interleaved loading of scenes from different datasets.
# It also reduces the memory consumption.
@lru_cache(maxsize=1)
def _decode_dataset_dto(dataset_path: AnyPath) -> DatasetDTO:
    dataset_cloud_path: AnyPath = dataset_path
    scene_dataset_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
    if not scene_dataset_json_path.exists():
        raise FileNotFoundError(f"File {scene_dataset_json_path} not found.")

    scene_dataset_json = read_json(path=scene_dataset_json_path)
    scene_dataset_dto = DatasetDTO.from_dict(scene_dataset_json)

    return scene_dataset_dto


class _DatasetDecoderMixin:
    def __init__(self, dataset_path: Union[str, AnyPath], **kwargs):
        self._dataset_path: AnyPath = AnyPath(dataset_path)

    def _decode_scene_paths(self) -> List[Path]:
        dto = self._decode_dataset_dto()
        return [
            Path(path)
            for split_key in sorted(dto.scene_splits.keys())
            for path in dto.scene_splits[split_key].filenames
        ]

    def _decode_dataset_dto(self) -> DatasetDTO:
        return _decode_dataset_dto(self._dataset_path)

    def _decode_scene_names(self) -> List[SceneName]:
        return [p.parent.name for p in self._decode_scene_paths()]


def decode_class_maps(
    ontologies: Dict[str, str], dataset_path: AnyPath, scene_name: SceneName
) -> Dict[AnnotationIdentifier, ClassMap]:
    decoded_ontologies = {}
    for annotation_key, ontology_file in ontologies.items():
        ontology_path = dataset_path / scene_name / "ontology" / f"{ontology_file}.json"
        ontology_data = read_json(path=ontology_path)

        ontology_dto = OntologyFileDTO.from_dict(ontology_data)
        decoded_ontologies[AnnotationIdentifier(annotation_type=ANNOTATION_TYPE_MAP[annotation_key])] = ClassMap(
            classes=[
                ClassDetail(
                    name=o.name,
                    id=o.id,
                    instanced=o.isthing,
                    meta={"color": {"r": o.color.r, "g": o.color.g, "b": o.color.b}},
                )
                for o in ontology_dto.items
            ]
        )

    return decoded_ontologies
