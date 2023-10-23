import logging
from typing import Iterable, List, Optional, Type, TypeVar, Union

import numpy as np
import pd.state
from google.protobuf.message import Message
from pd.core import PdError
from pd.internal.proto.keystone.generated.python import pd_sensor_pb2 as pd_sensor_pb2_base
from pd.internal.proto.keystone.generated.wrapper import pd_sensor_pb2
from pd.internal.proto.keystone.generated.wrapper.utils import register_wrapper
from pd.state.sensor import CameraSensor
from pd.state.sensor import DistortionParams as DistortionParamsStep
from pd.state.sensor import LiDARSensor
from pd.state.sensor import NoiseParams as NoiseParamsStep
from pd.state.sensor import PostProcessMaterial
from pd.state.sensor import PostProcessParams as PostProcessParamsStep

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.utilities import clip_with_warning, inherit_docs
from paralleldomain.utilities.coordinate_system import SIM_COORDINATE_SYSTEM, CoordinateSystem
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


DenoiseFilter = pd_sensor_pb2.DenoiseFilter


@inherit_docs
class AlbedoWeights(pd_sensor_pb2.AlbedoWeights):
    ...


@inherit_docs
class CameraIntrinsic(pd_sensor_pb2.CameraIntrinsic):
    ...


@inherit_docs
class DistortionParams(pd_sensor_pb2.DistortionParams):
    ...


@inherit_docs
class LidarBeam(pd_sensor_pb2.LidarBeam):
    ...


@inherit_docs
class LidarIntensityParams(pd_sensor_pb2.LidarIntensityParams):
    ...


@inherit_docs
class LidarIntrinsic(pd_sensor_pb2.LidarIntrinsic):
    ...


@inherit_docs
class LidarNoiseParams(pd_sensor_pb2.LidarNoiseParams):
    ...


@inherit_docs
class NoiseParams(pd_sensor_pb2.NoiseParams):
    ...


@inherit_docs
class PostProcessNode(pd_sensor_pb2.PostProcessNode):
    ...


@inherit_docs
class PostProcessParams(pd_sensor_pb2.PostProcessParams):
    ...


@inherit_docs
class RadarBasicParameters(pd_sensor_pb2.RadarBasicParameters):
    ...


@inherit_docs
class RadarDetectorParameters(pd_sensor_pb2.RadarDetectorParameters):
    ...


@inherit_docs
class RadarEnergyParameters(pd_sensor_pb2.RadarEnergyParameters):
    ...


@inherit_docs
class RadarIntrinsic(pd_sensor_pb2.RadarIntrinsic):
    ...


@inherit_docs
class RadarNoiseParameters(pd_sensor_pb2.RadarNoiseParameters):
    ...


@inherit_docs
class SensorExtrinsic(pd_sensor_pb2.SensorExtrinsic):
    ...


@inherit_docs
class SensorList(pd_sensor_pb2.SensorList):
    ...


@inherit_docs
class ToneCurveParams(pd_sensor_pb2.ToneCurveParams):
    ...


T = TypeVar("T")


def convert_to_step_class(step_class: Type[T], message: Message, attr_name: str) -> Optional[Union[T, List[T]]]:
    def _convert_message(nested_message: Message) -> T:
        target_vars = {k for k in vars(step_class) if not k.startswith("__") and not str(k).isupper()}
        return step_class(
            **{
                k: getattr(nested_message, k)
                for k in vars(nested_message.__class__)
                if not k.startswith("__") and not str(k).isupper() and k in target_vars
            }
        )

    if attr_name in [f[0].name for f in message.ListFields()]:
        sub_message = getattr(message, attr_name)
        if isinstance(sub_message, Iterable):
            submsgs = list()
            for msg in sub_message:
                submsgs.append(_convert_message(nested_message=msg))
            return submsgs
        else:
            return _convert_message(nested_message=sub_message)
    return None


@inherit_docs
@register_wrapper(proto_type=pd_sensor_pb2_base.SensorConfig)
class SensorConfig(pd_sensor_pb2.SensorConfig):
    @property
    def name(self) -> str:
        return self.display_name

    @property
    def sensor_to_ego(self) -> Transformation:
        """
        Returns:
            The transformation from sensor to ego in FLU
        """
        return self.get_sensor_to_ego()

    def get_sensor_to_ego(self, coordinate_system: str = "FLU") -> Transformation:
        """
        Args:
            coordinate_system: The coordinate system the return transformation is in

        Returns:
            The transformation from sensor to ego in the given coordinate system
        """
        extrinsic: SensorExtrinsic = self.sensor_extrinsic
        # sim coordinates are in  RFU, so roll is around the y-axis, pitch around x-axis
        pose_in_rfu = Transformation.from_euler_angles(
            angles=[extrinsic.proto.roll, extrinsic.proto.pitch, extrinsic.proto.yaw],
            order="yxz",
            translation=[extrinsic.proto.x, extrinsic.proto.y, extrinsic.proto.z],
            degrees=True,
        )
        return CoordinateSystem.change_transformation_coordinate_system(
            transformation=pose_in_rfu,
            transformation_system=SIM_COORDINATE_SYSTEM.axis_directions,
            target_system=coordinate_system,
        )

    @property
    def ego_to_sensor(self) -> Transformation:
        """
        Returns:
            The transformation from ego to sensor in FLU
        """
        return self.sensor_to_ego.inverse

    @property
    def is_camera(self) -> bool:
        return "camera_intrinsic" == self.proto.WhichOneof("sensor_intrinsic")

    @property
    def is_lidar(self) -> bool:
        return "lidar_intrinsic" == self.proto.WhichOneof("sensor_intrinsic")

    @property
    def is_radar(self) -> bool:
        return "radar_intrinsic" == self.proto.WhichOneof("sensor_intrinsic")

    @property
    def intrinsic_config(self) -> Union[LidarIntrinsic, RadarIntrinsic, CameraIntrinsic]:
        field_name = self.proto.WhichOneof("sensor_intrinsic")
        return getattr(self, field_name)

    @property
    def annotations_types(self) -> List[AnnotationType]:
        return get_annotations_types(intrinsics=self.intrinsic_config)

    def add_annotation_type(self, annotation_type: AnnotationType):
        add_annotation_type(intrinsics=self.intrinsic_config, annotation_type=annotation_type)

    def to_step_sensor(self) -> Union[CameraSensor, LiDARSensor]:
        sim_pose = self.get_sensor_to_ego(coordinate_system=SIM_COORDINATE_SYSTEM.axis_directions)
        sim_pose = pd.state.Pose6D.from_transformation_matrix(matrix=sim_pose.transformation_matrix)
        if self.is_camera:
            sensor = CameraSensor(
                name=self.display_name,
                pose=sim_pose,
                width=self.camera_intrinsic.width,
                height=self.camera_intrinsic.height,
                field_of_view_degrees=self.camera_intrinsic.fov,
                capture_rgb=self.camera_intrinsic.capture_rgb,
                capture_segmentation=self.camera_intrinsic.capture_segmentation,
                capture_depth=self.camera_intrinsic.capture_depth,
                capture_instances=self.camera_intrinsic.capture_instance,
                capture_normals=self.camera_intrinsic.capture_normals,
                capture_properties=self.camera_intrinsic.capture_properties,
                capture_basecolor=self.camera_intrinsic.capture_basecolor,
                capture_motionvectors=self.camera_intrinsic.capture_motionvectors,
                capture_backwardmotionvectors=self.camera_intrinsic.capture_backwardmotionvectors,
                supersample=self.camera_intrinsic.supersample,
                lut=self.camera_intrinsic.lut,
                lut_weight=self.camera_intrinsic.lut_weight,
                distortion_params=convert_to_step_class(
                    DistortionParamsStep, message=self.camera_intrinsic.proto, attr_name="distortion_params"
                ),
                noise_params=convert_to_step_class(
                    NoiseParamsStep, message=self.camera_intrinsic.proto, attr_name="noise_params"
                ),
                post_process_params=convert_to_step_class(
                    PostProcessParamsStep, message=self.camera_intrinsic.proto, attr_name="post_process_params"
                ),
                post_process_materials=convert_to_step_class(
                    PostProcessMaterial, message=self.camera_intrinsic.proto, attr_name="post_process"
                ),
                transmit_gray=self.camera_intrinsic.transmit_gray,
                fisheye_model=self.camera_intrinsic.distortion_params.fisheye_model,
                distortion_lookup_table=self.camera_intrinsic.distortion_lookup_table,
                time_offset=self.camera_intrinsic.time_offset,
            )
            sensor.render_ego = self.render_ego

            return sensor
        elif self.is_lidar:
            sensor = LiDARSensor(
                name=self.display_name,
                pose=sim_pose,
                azimuth_max=self.lidar_intrinsic.azimuth_max,
                azimuth_min=self.lidar_intrinsic.azimuth_min,
                # beam_data=self.lidar_intrinsic.beam_data,
                # Deactivated until Data Lab supports LiDAR  annotations
                # capture_backwardmotionvectors=self.camera_intrinsic.capture_backwardmotionvectors,
                # capture_depth=self.lidar_intrinsic.capture_depth,
                # capture_detections=self.lidar_intrinsic.capture_detections,
                # capture_instances=self.lidar_intrinsic.capture_instance,
                # capture_motionvectors=self.lidar_intrinsic.capture_motionvectors,
                # capture_normals=self.lidar_intrinsic.capture_normals,
                # capture_rgb=self.lidar_intrinsic.capture_rgb,
                # capture_segmentation=self.lidar_intrinsic.capture_segmentation,
                # intensity_params=self.lidar_intrinsic.intensity_params,
                maximum_cutoff_prob=self.lidar_intrinsic.maximum_cutoff_prob,
                maximum_offset=self.lidar_intrinsic.maximum_offset,
                maximum_range_cutoff=self.lidar_intrinsic.maximum_range_cutoff,
                minimum_cutoff_prob=self.lidar_intrinsic.minimum_cutoff_prob,
                minimum_noise=self.lidar_intrinsic.minimum_noise,
                minimum_offset=self.lidar_intrinsic.minimum_offset,
                minimum_range_cutoff=self.lidar_intrinsic.minimum_range_cutoff,
                pattern=self.lidar_intrinsic.pattern,
                range_noise_mean=self.lidar_intrinsic.range_noise_mean,
                range_noise_stddev=self.lidar_intrinsic.range_noise_stddev,
                rotation_rate=self.lidar_intrinsic.rotation_rate,
                sample_rate=self.lidar_intrinsic.sample_rate,
                time_offset_ms=self.lidar_intrinsic.time_offset,
            )
            sensor.render_ego = self.render_ego

            return sensor
        else:
            raise NotImplementedError()

    @staticmethod
    def create_camera_sensor(
        name: str,
        pose: Union[Transformation, np.ndarray],
        width: int,
        height: int,
        field_of_view_degrees: float,
        grayscale: bool = False,
        follow_rotation: bool = True,
        lock_to_yaw: bool = False,
        annotation_types: List[AnnotationType] = None,
        pose_coordinate_system: str = SIM_COORDINATE_SYSTEM.axis_directions,
        **kwargs,
    ) -> "SensorConfig":
        if isinstance(pose, np.ndarray):
            pose = Transformation.from_transformation_matrix(pose)
        pose = CoordinateSystem.change_transformation_coordinate_system(
            transformation=pose,
            transformation_system=pose_coordinate_system,
            target_system=SIM_COORDINATE_SYSTEM.axis_directions,
        )

        x, y, z = pose.translation
        # RFU coordinate system => roll is around y-axis, pitch around x-axis
        roll, pitch, yaw = pose.as_euler_angles(order="yxz", degrees=True)
        extrinsic = SensorExtrinsic(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, follow_rotation=follow_rotation, lock_to_yaw=lock_to_yaw
        )
        annotation_types = annotation_types if annotation_types is not None else []
        return SensorConfig(
            display_name=name,
            render_ego=kwargs.pop("render_ego", False),
            sensor_extrinsic=extrinsic,
            camera_intrinsic=CameraIntrinsic(
                width=width,
                height=height,
                fov=field_of_view_degrees,
                capture_rgb=True,
                capture_segmentation=AnnotationTypes.SemanticSegmentation2D in annotation_types,
                capture_depth=AnnotationTypes.Depth in annotation_types,
                capture_instance=AnnotationTypes.InstanceSegmentation2D in annotation_types,
                capture_normals=AnnotationTypes.SurfaceNormals2D in annotation_types,
                capture_properties=AnnotationTypes.MaterialProperties2D in annotation_types,
                capture_basecolor=AnnotationTypes.Albedo2D in annotation_types,
                capture_motionvectors=AnnotationTypes.OpticalFlow in annotation_types,
                capture_backwardmotionvectors=AnnotationTypes.BackwardOpticalFlow in annotation_types,
                transmit_gray=grayscale,
                **kwargs,
            ),
        )


@inherit_docs
@register_wrapper(proto_type=pd_sensor_pb2_base.SensorRigConfig)
class SensorRig(pd_sensor_pb2.SensorRigConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._validate_sensor_config()

    def _validate_sensor_config(self):
        maximum_megapixels = 64_000_000  # 64 megapixels
        total_pixel_count = 0

        for sensor in self.sensors:
            if sensor.is_camera:
                distortion_params = sensor.intrinsic_config.distortion_params
                fisheye_model = distortion_params.fisheye_model
                width = sensor.intrinsic_config.width
                height = sensor.intrinsic_config.height

                # There is a maximum camera height and width specified in the IG
                if width > 4200 or height > 4200:
                    raise PdError(
                        f"Camera '{sensor.name}' has a height or width greater than 4200 pixels. Reduce the resolution"
                        " and submit again."
                    )
                if width <= 1 or height <= 1:
                    raise PdError(
                        f"Camera '{sensor.name}' has a height or width less than 2 pixels. Increase the resolution and"
                        " submit again."
                    )

                if fisheye_model == 6 and distortion_params.p1 < -300:
                    distortion_params.p1 = -300
                    logger.warning(
                        f"Clipping near clip plane of camera '{sensor.name}' to -300m. This is the minimum value"
                        " currently supported."
                    )

                if fisheye_model == 6 and distortion_params.p2 > 500:
                    distortion_params.p2 = 500
                    logger.warning(
                        f"Clipping far clip plane of camera '{sensor.name}' to 500m. This is the maximum value"
                        " currently supported."
                    )

                if fisheye_model == 6 and distortion_params.p1 > distortion_params.p2:
                    raise PdError(
                        f"Camera '{sensor.name}' is an orthographic camera with a near clip plane"
                        f" ({distortion_params.p1}m) further away than its far clip plane"
                        f" ({distortion_params.p2}m). This is not supported."
                    )

                if fisheye_model not in {0, 1, 3, 6}:
                    raise PdError(
                        f"Camera '{sensor.name}' has an unsupported fisheye model. Only models 0, 1, 3, and 6 are"
                        " supported."
                    )

                if sensor.intrinsic_config.fov == 0.0 and distortion_params.fx == 0.0:
                    raise PdError(
                        f"Camera '{sensor.name}' has a field of view of 0.0 degrees and a focal length of 0.0 pixels."
                        " Please specify a field of view or focal length."
                    )

                pixel_factor = 3.0 if fisheye_model in [1, 3, 6] else 1.0
                supersample_factor = (
                    1.0 if sensor.intrinsic_config.supersample == 0.0 else sensor.intrinsic_config.supersample
                )  # The supersample defaults to 0.0 when not set, but is handled by the IG.  We handle this case

                total_pixel_count += pixel_factor * supersample_factor * width * height

        if total_pixel_count > maximum_megapixels:
            raise PdError(
                "The specified camera configuration is too large, please reduce the number of cameras or the combined"
                " total resolution of all cameras, including supersampling factors."
            )

    @property
    def available_annotations(self) -> List[AnnotationType]:
        available_annotations = set()
        for sensor in self.sensors:
            for annotation in sensor.annotations_types:
                available_annotations.add(annotation)
        return list(available_annotations)

    @property
    def sensors(self) -> List[SensorConfig]:
        return list(self.sensor_configs)

    @property
    def sensor_names(self) -> List[str]:
        return [s.name for s in self.sensors]

    @property
    def camera_names(self) -> List[str]:
        return [s.name for s in self.cameras]

    @property
    def lidar_names(self) -> List[str]:
        return [s.name for s in self.lidars]

    @property
    def cameras(self) -> List[SensorConfig]:
        return [s for s in self.sensors if s.is_camera]

    @property
    def lidars(self) -> List[SensorConfig]:
        return [s for s in self.sensors if s.is_lidar]

    def add_sensor(self, sensor: SensorConfig):
        if sensor.name in self.sensor_names:
            raise ValueError(f"Sensor with name {sensor.name} already exists in this Rig!")

        self.sensor_configs.append(sensor)

    def add_camera(
        self,
        name: str,
        pose: Union[Transformation, np.ndarray],
        width: int,
        height: int,
        field_of_view_degrees: float = 70,
        grayscale: bool = False,
        annotation_types: List[AnnotationType] = None,
        **kwargs,
    ) -> "SensorRig":
        cam = SensorConfig.create_camera_sensor(
            name=name,
            pose=pose,
            width=width,
            height=height,
            field_of_view_degrees=field_of_view_degrees,
            grayscale=grayscale,
            annotation_types=annotation_types,
            **kwargs,
        )
        self.add_sensor(sensor=cam)

        self._validate_sensor_config()

        return self


def get_annotations_types(intrinsics: Union[LidarIntrinsic, RadarIntrinsic, CameraIntrinsic]) -> List[AnnotationType]:
    anno_types = list()
    if intrinsics.proto.capture_segmentation:
        anno_types.append(AnnotationTypes.SemanticSegmentation2D)
    if intrinsics.proto.capture_depth:
        anno_types.append(AnnotationTypes.Depth)
    if intrinsics.proto.capture_instance:
        anno_types.append(AnnotationTypes.InstanceSegmentation2D)
    if intrinsics.proto.capture_normals:
        anno_types.append(AnnotationTypes.SurfaceNormals2D)
    if intrinsics.proto.capture_properties:
        anno_types.append(AnnotationTypes.MaterialProperties2D)
    if isinstance(intrinsics, CameraIntrinsic) and intrinsics.proto.capture_basecolor:
        anno_types.append(AnnotationTypes.Albedo2D)
    if intrinsics.proto.capture_motionvectors:
        anno_types.append(AnnotationTypes.OpticalFlow)
    if intrinsics.proto.capture_backwardmotionvectors:
        anno_types.append(AnnotationTypes.BackwardOpticalFlow)
    return anno_types


def add_annotation_type(
    intrinsics: Union[LidarIntrinsic, RadarIntrinsic, CameraIntrinsic], annotation_type: AnnotationType
):
    if AnnotationTypes.SemanticSegmentation2D == annotation_type:
        intrinsics.capture_segmentation = True
    elif AnnotationTypes.Depth == annotation_type:
        intrinsics.capture_depth = True
    elif AnnotationTypes.InstanceSegmentation2D == annotation_type:
        intrinsics.capture_instance = True
    elif AnnotationTypes.SurfaceNormals2D == annotation_type:
        intrinsics.capture_normals = True
    elif AnnotationTypes.MaterialProperties2D == annotation_type:
        intrinsics.capture_properties = True
    elif AnnotationTypes.Albedo2D == annotation_type:
        intrinsics.capture_basecolor = True
    elif AnnotationTypes.OpticalFlow == annotation_type:
        intrinsics.capture_motionvectors = True
    elif AnnotationTypes.BackwardOpticalFlow == annotation_type:
        intrinsics.capture_backwardmotionvectors = True
