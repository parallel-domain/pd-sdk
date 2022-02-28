from typing import Any, Optional

from paralleldomain.encoding.dgp.v1.format.bounding_box_2d import BoundingBox2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.bounding_box_3d import BoundingBox3DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.camera_image import CameraDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.class_map import ClassMapDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.dataset import DatasetDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.depth import DepthDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.instance_segmentation_2d import InstanceSegmentation2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.instance_segmentation_3d import InstanceSegmentation3DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.optical_flow import OpticalFlowDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.point_cloud import PointCloudDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.points_2d import Point2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.polygons_2d import Polygons2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.polylines_2d import Polyline2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.scene import SceneDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.scene_flow import SceneFlowDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.semantic_segmentation_2d import SemanticSegmentation2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.semantic_segmentation_3d import SemanticSegmentation3DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.surface_normals_2d import SurfaceNormals2DDGPV1Mixin
from paralleldomain.encoding.dgp.v1.format.surface_normals_3d import SurfaceNormals3DDGPV1Mixin
from paralleldomain.encoding.pipeline_encoder import DataType, EncodingFormat, ScenePipelineItem, SensorDataTypes
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.utilities.any_path import AnyPath


class DGPV1EncodingFormat(
    EncodingFormat[ScenePipelineItem],
    BoundingBox2DDGPV1Mixin,
    CameraDGPV1Mixin,
    SceneDGPV1Mixin,
    DatasetDGPV1Mixin,
    BoundingBox3DDGPV1Mixin,
    DepthDGPV1Mixin,
    InstanceSegmentation2DDGPV1Mixin,
    InstanceSegmentation3DDGPV1Mixin,
    OpticalFlowDGPV1Mixin,
    PointCloudDGPV1Mixin,
    Point2DDGPV1Mixin,
    Polygons2DDGPV1Mixin,
    Polyline2DDGPV1Mixin,
    SceneFlowDGPV1Mixin,
    SemanticSegmentation2DDGPV1Mixin,
    SemanticSegmentation3DDGPV1Mixin,
    SurfaceNormals2DDGPV1Mixin,
    SurfaceNormals3DDGPV1Mixin,
    ClassMapDGPV1Mixin,
):
    def __init__(
        self,
        dataset_output_path: AnyPath,
        target_dataset_name: Optional[str],
        sim_offset: float = 0.01 * 5,
    ):
        super().__init__()
        self.target_dataset_name = target_dataset_name
        self.sim_offset = sim_offset
        self.dataset_output_path = dataset_output_path

    def supports_copy(self, pipeline_item: ScenePipelineItem, data_type: DataType, data_path: AnyPath):
        if pipeline_item.dataset_format == "dgpv1":
            return True
        elif pipeline_item.dataset_format == "dgp":
            if data_type in [
                Image,
                AnnotationTypes.Depth,
                AnnotationTypes.SemanticSegmentation2D,
                AnnotationTypes.SemanticSegmentation3D,
                AnnotationTypes.InstanceSegmentation2D,
                AnnotationTypes.InstanceSegmentation3D,
                AnnotationTypes.SceneFlow,
                AnnotationTypes.OpticalFlow,
                AnnotationTypes.SurfaceNormals3D,
            ]:
                return True
        return False

    def save_data(self, pipeline_item: ScenePipelineItem, data_type: DataType, data: Any):
        self.ensure_format_data_exists(pipeline_item=pipeline_item)
        scene_output_path = self.dataset_output_path / pipeline_item.scene_name

        if data_type == AnnotationTypes.BoundingBoxes2D:
            self.save_boxes_2d_and_write_state(
                pipeline_item=pipeline_item,
                data=data,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.BoundingBoxes3D:
            self.save_boxes_3d_and_write_state(
                pipeline_item=pipeline_item,
                data=data,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == Image:
            self.save_image_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.Depth:
            self.save_depth_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.InstanceSegmentation2D:
            self.save_instance_segmentation_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.InstanceSegmentation3D:
            self.save_instance_segmentation_3d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.OpticalFlow:
            self.save_optical_flow_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == PointCloud:
            self.save_point_cloud_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.Points2D:
            self.save_points_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.Polygons2D:
            self.save_polygons_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.Polylines2D:
            self.save_polyline_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.SceneFlow:
            self.save_scene_flow_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.SemanticSegmentation2D:
            self.save_semantic_segmentation_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.SemanticSegmentation3D:
            self.save_semantic_segmentation_3d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.SurfaceNormals2D:
            self.save_surface_normals_2d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == AnnotationTypes.SurfaceNormals3D:
            self.save_surface_normals_3d_and_write_state(
                data=data,
                pipeline_item=pipeline_item,
                scene_output_path=scene_output_path,
                sim_offset=self.sim_offset,
            )
        elif data_type == ClassMap:
            self.save_class_map_in_state(
                data=data,
                pipeline_item=pipeline_item,
            )

    def save_sensor_frame(self, pipeline_item: ScenePipelineItem, data: Any = None):
        self.aggregate_sensor_frame(pipeline_item=pipeline_item)

    def save_scene(self, pipeline_item: ScenePipelineItem, data: Any = None):
        self.save_aggregated_scene(pipeline_item=pipeline_item, dataset_output_path=self.dataset_output_path)

    def save_dataset(self, pipeline_item: ScenePipelineItem, data: Any = None):
        self.save_aggregated_dataset(
            pipeline_item=pipeline_item,
            dataset_output_path=self.dataset_output_path,
            target_dataset_name=self.target_dataset_name,
        )
