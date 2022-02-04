import logging
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from paralleldomain import Dataset, Scene
from paralleldomain.encoding.dgp.v1.encoder_steps.bounding_boxes_2d import BoundingBoxes2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.bounding_boxes_3d import BoundingBoxes3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.camera_image import CameraImageEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.depth import DepthEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.instance_segmentation_2d import InstanceSegmentation2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.instance_segmentation_3d import InstanceSegmentation3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.optical_flow import OpticalFlowEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.point_cloud import PointCloudEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.points_2d import Points2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.polygons_2d import Polygons2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.polylines_2d import Polylines2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.scene import SceneEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.scene_aggregator import DGPV1SceneAggregator
from paralleldomain.encoding.dgp.v1.encoder_steps.scene_flow import SceneFlowEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.semantic_segmentation_2d import SemanticSegmentation2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.semantic_segmentation_3d import SemanticSegmentation3DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.surface_normals_2d import SurfaceNormals2DEncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.surface_normals_3d import SurfaceNormals3DEncoderStep
from paralleldomain.encoding.pipeline_encoder import EncoderStep, FinalStep, PipelineBuilder, S, SceneAggregator, T
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import CameraSensor, LidarSensor
from paralleldomain.model.type_aliases import FrameId, SceneName
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPV1PipelineBuilder(PipelineBuilder[Scene, Dict[str, Any]]):
    def __init__(
        self,
        output_path: AnyPath,
        encoder_steps_builder: Optional[Callable[[], List[EncoderStep]]] = None,
        final_encoder_step_builder: Optional[Callable[[Tuple[SceneName, str]], FinalStep]] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        sim_offset: float = 0.01 * 5,
        stages_max_out_queue_size: int = 3,
        workers_per_step: int = 2,
        max_queue_size_per_step: int = 4,
        max_queue_size_final_step: int = 20,
        allowed_frames: Optional[List[FrameId]] = None,
        output_annotation_types: Optional[List[AnnotationType]] = None,
        target_dataset_name: Optional[str] = None,
    ):

        self.output_annotation_types = output_annotation_types
        self.target_dataset_name = target_dataset_name
        self.allowed_frames = allowed_frames
        if encoder_steps_builder is None:
            encoder_steps_builder = partial(
                DGPV1PipelineBuilder.get_default_encoder_steps,
                workers_per_step=workers_per_step,
                max_queue_size_per_step=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            )
        if final_encoder_step_builder is None:
            final_encoder_step_builder = partial(
                DGPV1PipelineBuilder.get_default_final_step, max_queue_size=max_queue_size_final_step
            )

        self.final_encoder_step_builder = final_encoder_step_builder
        self.stages_max_out_queue_size = stages_max_out_queue_size
        self.encoder_steps_builder = encoder_steps_builder
        self.sensor_names = sensor_names
        self.sim_offset = sim_offset
        self.output_path = output_path

    def build_scene_aggregator(self, dataset: Dataset) -> SceneAggregator[Dict[str, Any]]:
        name = dataset.name if self.target_dataset_name is None else self.target_dataset_name
        return DGPV1SceneAggregator(output_path=self.output_path, dataset_name=name)

    def build_scene_encoder_steps(self, dataset: Dataset, scene: Scene) -> List[EncoderStep]:
        return self.encoder_steps_builder()

    def build_scene_final_encoder_step(self, dataset: Dataset, scene: Scene) -> FinalStep[Dict[str, Any]]:
        return self.final_encoder_step_builder(scene.name, scene.description)

    def build_pipeline_source_generator(self, dataset: Dataset, scene: Scene) -> Generator[Dict[str, Any], None, None]:
        if self.sensor_names is None:
            sensor_name_mapping = {s: s for s in scene.sensor_names}
        elif isinstance(self.sensor_names, list):
            sensor_name_mapping = {s: s for s in self.sensor_names if s in scene.sensor_names}
        elif isinstance(self.sensor_names, dict):
            sensor_name_mapping = {t: s for t, s in self.sensor_names.items() if s in scene.sensor_names}
        else:
            raise ValueError(f"sensor_names is neither a list nor a dict but {type(self.sensor_names)}!")

        reference_timestamp: datetime = scene.get_frame(scene.frame_ids[0]).date_time
        output_path = self.output_path / scene.name

        logger.info(f"Encoding Scene {scene.name} with sensor mapping: {sensor_name_mapping}")
        for target_sensor_name, source_sensor_name in sensor_name_mapping.items():
            sensor = scene.get_sensor(sensor_name=source_sensor_name)
            if sensor.name in sensor_name_mapping:
                for sensor_frame in sensor.sensor_frames:
                    if self.allowed_frames is None or sensor_frame.frame_id in self.allowed_frames:
                        if isinstance(sensor, CameraSensor):
                            yield dict(
                                camera_frame_info=dict(
                                    sensor_name=sensor.name,
                                    frame_id=sensor_frame.frame_id,
                                    scene_name=scene.name,
                                    dataset_path=dataset.path,
                                    dataset_format=dataset.format,
                                    decoder_kwargs=dataset.decoder_init_kwargs,
                                ),
                                target_sensor_name=target_sensor_name,
                                scene_output_path=output_path,
                                scene_reference_timestamp=reference_timestamp,
                                sim_offset=self.sim_offset,
                            )
                        elif isinstance(sensor, LidarSensor):
                            yield dict(
                                lidar_frame_info=dict(
                                    sensor_name=sensor.name,
                                    frame_id=sensor_frame.frame_id,
                                    scene_name=scene.name,
                                    dataset_path=dataset.path,
                                    dataset_format=dataset.format,
                                    decoder_kwargs=dataset.decoder_init_kwargs,
                                ),
                                target_sensor_name=target_sensor_name,
                                scene_output_path=output_path,
                                scene_reference_timestamp=reference_timestamp,
                                sim_offset=self.sim_offset,
                            )

    @staticmethod
    def get_default_encoder_steps(
        workers_per_step: int = 2,
        max_queue_size_per_step: int = 4,
        output_annotation_types: Optional[List[AnnotationType]] = None,
    ) -> List[EncoderStep]:
        encoders = [
            BoundingBoxes2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
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
                output_annotation_types=output_annotation_types,
            ),
            InstanceSegmentation2DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SemanticSegmentation3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            InstanceSegmentation3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            DepthEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            OpticalFlowEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            Points2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            Polygons2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            Polylines2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SceneFlowEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SurfaceNormals2DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SurfaceNormals3DEncoderStep(
                fs_copy=True,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            BoundingBoxes3DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
        ]
        return encoders

    @staticmethod
    def get_default_final_step(
        target_scene_name: SceneName,
        target_scene_description: str,
        max_queue_size: int = 20,
    ) -> FinalStep:
        return SceneEncoderStep(
            in_queue_size=max_queue_size,
            target_scene_name=target_scene_name,
            target_scene_description=target_scene_description,
        )
