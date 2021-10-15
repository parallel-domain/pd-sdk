from datetime import datetime
from math import modf
from typing import Any, Dict, List

import dataclasses_json
import ujson
from dataclasses_json import DataClassJsonMixin
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.pyext._message import Message

from paralleldomain.common.dgp.v1 import ontology_pb2
from paralleldomain.model.class_mapping import ClassMap


def class_map_to_ontology_proto(class_map: ClassMap):
    return ontology_pb2.Ontology(
        items=[
            ontology_pb2.OntologyItem(
                id=cid,
                name=cval.name,
                color=ontology_pb2.OntologyItem.Color(
                    r=cval.meta["color"]["r"],
                    g=cval.meta["color"]["g"],
                    b=cval.meta["color"]["b"],
                ),
                isthing=cval.instanced,
                supercategory="",
            )
            for cid, cval in class_map.items()
        ]
    )


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return ujson.dumps(obj, indent=2, escape_forward_slashes=False)
    else:
        return str(obj)
