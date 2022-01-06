from json import JSONDecodeError
from typing import Any, Dict, TypeVar, Union

import ujson


class NodePrefix:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


T = TypeVar("T")


def load_user_data(user_data: T) -> Union[T, Dict[str, Any]]:
    try:
        return ujson.loads(user_data)
    except (ValueError, JSONDecodeError):
        return user_data
