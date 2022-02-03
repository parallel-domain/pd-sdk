from typing import List

from paralleldomain.encoding.dgp.v1.encoder_steps.bounding_boxes_2d import BoundingBoxes2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.bounding_boxes_3d import BoundingBoxes3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.camera_image import CameraImageEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.depth import DepthEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.instance_segmentation_2d import InstanceSegmentation2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.instance_segmentation_3d import InstanceSegmentation3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.optical_flow import OpticalFlowEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.point_cloud import PointCloudEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.points_2d import Points2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.polygons_2d import Polygons2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.polylines_2d import Polylines2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.scene_flow import SceneFlowEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.semantic_segmentation_2d import SemanticSegmentation2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.semantic_segmentation_3d import SemanticSegmentation3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.surface_normals_2d import SurfaceNormals2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.surface_normals_3d import SurfaceNormals3DEncoderStep
from paralleldomain.utilities.any_path import AnyPath


class EncoderSteps(List[EncoderStep]):
    @staticmethod
    def get_default(workers_per_step: int = 2, max_queue_size_per_step: int = 4) -> List[EncoderStep]:
        encoders = [
            BoundingBoxes2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            BoundingBoxes3DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            CameraImageEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            PointCloudEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SemanticSegmentation2DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            InstanceSegmentation2DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SemanticSegmentation3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            InstanceSegmentation3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            DepthEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            OpticalFlowEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            # Points2DEncoderStep(
            #     workers=workers_per_step,
            #     in_queue_size=max_queue_size_per_step,
            # ),
            Polygons2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            Polylines2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SceneFlowEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SurfaceNormals2DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SurfaceNormals3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
        ]
        return encoders
