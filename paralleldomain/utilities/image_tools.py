from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import rasterio.features


class FeatureConnectivity(Enum):
    VON_NEUMANN = 4
    MOORE = 8


def mask_to_polygons(
    mask: np.ndarray, connectivity: FeatureConnectivity = FeatureConnectivity.VON_NEUMANN
) -> List[Union[Dict[str, Any], float]]:
    shapes = rasterio.features.shapes(mask.astype("int32"), connectivity=connectivity.value)

    return [p for p in shapes if p[0]["type"] == "Polygon"]
