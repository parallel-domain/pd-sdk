from typing import Dict

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.common.dgp.v0.dtos import OntologyFileDTO, SceneDTO
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json


def decode_class_maps(
    ontologies: Dict[str, str], dataset_path: AnyPath, scene_name: SceneName
) -> Dict[AnnotationType, ClassMap]:
    decoded_ontologies = {}
    for annotation_key, ontology_file in ontologies.items():
        ontology_path = dataset_path / scene_name / "ontology" / f"{ontology_file}.json"
        ontology_data = read_json(path=ontology_path)

        ontology_dto = OntologyFileDTO.from_dict(ontology_data)
        decoded_ontologies[ANNOTATION_TYPE_MAP[annotation_key]] = ClassMap(
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
