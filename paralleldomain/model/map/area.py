from dataclasses import dataclass
from typing import Optional

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.type_aliases import AreaId


@dataclass
class Area:
    area_id: AreaId
    bounds: Optional[BoundingBox2DGeometry]
