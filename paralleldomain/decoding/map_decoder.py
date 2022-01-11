import abc
from typing import Dict, List, Optional

from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.edge import Edge
from paralleldomain.model.map.junction import Junction
from paralleldomain.model.map.lane_segment import LaneSegment
from paralleldomain.model.map.road_segment import RoadSegment
from paralleldomain.model.type_aliases import (
    AreaId,
    EdgeId,
    FrameId,
    JunctionId,
    LaneSegmentId,
    RoadSegmentId,
    SceneName,
    SensorName,
)


class MapDecoder(LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings):
        self.scene_name = scene_name
        self.settings = settings
        self.dataset_name = dataset_name

    def get_unique_id(
        self,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    @abc.abstractmethod
    def decode_road_segments(self) -> Dict[RoadSegmentId, RoadSegment]:
        pass

    @abc.abstractmethod
    def decode_lane_segments(self) -> Dict[LaneSegmentId, LaneSegment]:
        pass

    @abc.abstractmethod
    def decode_junctions(self) -> Dict[JunctionId, Junction]:
        pass

    @abc.abstractmethod
    def decode_areas(self) -> Dict[AreaId, Area]:
        pass

    @abc.abstractmethod
    def decode_edges(self) -> Dict[EdgeId, Edge]:
        pass

    def get_road_segments(self) -> Dict[RoadSegmentId, RoadSegment]:
        _unique_cache_key = self.get_unique_id(extra="road_segments")
        road_segments = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self.decode_road_segments(),
        )
        return road_segments

    def get_lane_segments(self) -> Dict[LaneSegmentId, LaneSegment]:
        _unique_cache_key = self.get_unique_id(extra="lane_segments")
        lane_segments = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self.decode_lane_segments(),
        )
        return lane_segments

    def get_junctions(self) -> Dict[JunctionId, Junction]:
        _unique_cache_key = self.get_unique_id(extra="junctions")
        junctions = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self.decode_junctions(),
        )
        return junctions

    def get_areas(self) -> Dict[AreaId, Area]:
        _unique_cache_key = self.get_unique_id(extra="areas")
        areas = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self.decode_areas(),
        )
        return areas

    def get_edges(self) -> Dict[EdgeId, Edge]:
        _unique_cache_key = self.get_unique_id(extra="edges")
        edges = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self.decode_edges(),
        )
        return edges

    @abc.abstractmethod
    def get_lane_segments_from_points(self, poses: List[Point3DGeometry]) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segments_near_point(self, point: Point3DGeometry) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_road_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[Area]:
        pass
