from typing import Dict, List

import dataclasses_json
import ujson
from dataclasses_json import DataClassJsonMixin


class SkipNoneMixin(DataClassJsonMixin):
    dataclass_json_config = dataclasses_json.config(
        exclude=lambda f: f is None,
    )["dataclasses_json"]


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return ujson.dumps(obj, indent=2, escape_forward_slashes=False)
    else:
        return str(obj)
