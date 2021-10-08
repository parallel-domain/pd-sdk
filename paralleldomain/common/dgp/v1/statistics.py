from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin


@dataclass
class ImageStatisticsDTO(DataClassDictMixin):
    count: int
    mean: List[float]
    stddev: List[float]


@dataclass
class PointCloudStatisticsDTO(DataClassDictMixin):
    count: int
    mean: List[float]
    stddev: List[float]


@dataclass
class DatasetStatisticsDTO(DataClassDictMixin):
    image_statistics: ImageStatisticsDTO
    pointcloud_statistics: PointCloudStatisticsDTO
