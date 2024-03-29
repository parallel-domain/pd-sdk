from typing import Dict, Any

import ujson

from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP
from paralleldomain.common.dgp.v1.src.dgp.proto import ontology_pb2
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_message


def decode_class_maps(
    ontologies: Dict[str, str], dataset_path: AnyPath, scene_name: SceneName
) -> Dict[AnnotationIdentifier, ClassMap]:
    decoded_ontologies = {}
    for annotation_key, ontology_file in ontologies.items():
        ontology_path = dataset_path / scene_name / "ontology" / f"{ontology_file}.json"
        if not ontology_path.exists():
            ontology_path = dataset_path / scene_name / "ontology" / f"{ontology_file}.bin"
        ontology_dto = read_message(obj=ontology_pb2.Ontology(), path=ontology_path)

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


def map_container_to_dict(attributes: Dict[str, str]) -> Dict[str, Any]:
    attributes_decoded = {}
    for k, v in attributes.items():
        try:
            attributes_decoded[k] = ujson.loads(v)
        except (TypeError, ValueError, ujson.JSONDecodeError):
            attributes_decoded[k] = v

    return attributes_decoded
