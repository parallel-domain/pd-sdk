import logging
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.helper import decode_dataset
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
from paralleldomain.encoding.pipeline_encoder import EncoderStep, PipelineBuilder
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import CameraSensor, LidarSensor
from paralleldomain.model.type_aliases import FrameId, SceneName
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPV1PipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_format: str,
        output_path: AnyPath,
        inplace: bool = False,
        encoder_steps: List[EncoderStep] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        sim_offset: float = 0.01 * 5,
        stages_max_out_queue_size: int = 3,
        workers_per_step: int = 2,
        max_queue_size_per_step: int = 4,
        max_queue_size_final_step: int = 20,
        allowed_frames: Optional[List[FrameId]] = None,
        output_annotation_types: Optional[List[AnnotationType]] = None,
        target_dataset_name: Optional[str] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        fs_copy: bool = True,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.inplace = inplace
        self.max_queue_size_final_step = max_queue_size_final_step
        self.output_annotation_types = output_annotation_types
        self.target_dataset_name = target_dataset_name
        self.allowed_frames = allowed_frames
        self.sim_offset = sim_offset
        self.output_path = output_path
        self.sensor_names = sensor_names
        self.stages_max_out_queue_size = stages_max_out_queue_size

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        if "dataset_path" in decoder_kwargs:
            decoder_kwargs.pop("dataset_path")
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)

        self._dataset = dataset
        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

        if encoder_steps is None:
            encoder_steps = self.get_default_encoder_steps(
                workers_per_step=workers_per_step,
                max_queue_size_per_step=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
                fs_copy=fs_copy,
            )
        self.encoder_steps = encoder_steps

    def build_encoder_steps(self) -> List[EncoderStep]:
        return self.encoder_steps

    def build_pipeline_source_generator(self) -> Generator[Dict[str, Any], None, None]:
        dataset = self._dataset
        for scene_name in self._scene_names:
            scene = self._dataset.get_unordered_scene(scene_name=scene_name)
            # yield self._dataset, scene, dict()

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
                if sensor.name in sensor_name_mapping.values():
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

            yield dict(
                scene_info=dict(
                    scene_name=scene.name,
                    dataset_path=dataset.path,
                    dataset_format=dataset.format,
                    decoder_kwargs=dataset.decoder_init_kwargs,
                ),
                scene_output_path=output_path,
                scene_reference_timestamp=reference_timestamp,
                sim_offset=self.sim_offset,
                end_of_scene=True,
                target_scene_name=scene.name,
                target_scene_description=scene.description,
            )

    @property
    def pipeline_item_unit_name(self):
        return "sensor frames"

    def get_default_encoder_steps(
        self,
        workers_per_step: int = 2,
        max_queue_size_per_step: int = 4,
        output_annotation_types: Optional[List[AnnotationType]] = None,
        fs_copy: bool = True,
    ) -> List[EncoderStep]:
        encoders = [
            BoundingBoxes2DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            CameraImageEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            PointCloudEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
            ),
            SemanticSegmentation2DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            InstanceSegmentation2DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SemanticSegmentation3DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            InstanceSegmentation3DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            DepthEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            OpticalFlowEncoderStep(
                fs_copy=fs_copy,
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
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SurfaceNormals2DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SurfaceNormals3DEncoderStep(
                fs_copy=fs_copy,
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            BoundingBoxes3DEncoderStep(
                workers=workers_per_step,
                in_queue_size=max_queue_size_per_step,
                output_annotation_types=output_annotation_types,
            ),
            SceneEncoderStep(in_queue_size=self.max_queue_size_final_step, inplace=self.inplace),
            DGPV1SceneAggregator(
                output_path=self.output_path,
                dataset_name=self._dataset.name if self.target_dataset_name is None else self.target_dataset_name,
            ),
        ]
        return encoders
