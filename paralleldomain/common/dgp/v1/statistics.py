from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ImageStatisticsDTO:
    count: int
    mean: List[float]
    stddev: List[float]


@dataclass_json
@dataclass
class PointCloudStatisticsDTO:
    count: int
    mean: List[float]
    stddev: List[float]


@dataclass_json
@dataclass
class DatasetStatisticsDTO:
    image_statistics: ImageStatisticsDTO
    pointcloud_statistics: PointCloudStatisticsDTO
