import numpy as np
from typing import List, Dict, Union, Any
import rasterio.features
from enum import Enum


class FeatureConnectivity(Enum):
    VON_NEUMANN = 4
    MOORE = 8


def mask_to_polygons(
    mask: np.ndarray, connectivity: FeatureConnectivity = FeatureConnectivity.VON_NEUMANN
) -> List[Union[Dict[str, Any], float]]:
    shapes = rasterio.features.shapes(mask.astype("int32"), connectivity=connectivity.value)

    return [p for p in shapes if p[0]["type"] == "Polygon"]
