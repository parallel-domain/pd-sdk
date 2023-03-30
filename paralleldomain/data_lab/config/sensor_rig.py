from typing import Iterable, List, Optional, Type, TypeVar, Union

import numpy as np
import pd.state
from google.protobuf.message import Message
from pd.internal.proto.keystone.generated.python import pd_sensor_pb2 as pd_sensor_pb2_base
from pd.internal.proto.keystone.generated.wrapper import pd_sensor_pb2
from pd.internal.proto.keystone.generated.wrapper.utils import register_wrapper
from pd.state.sensor import (
    CameraSensor,
    DistortionParams,
    LiDARSensor,
    NoiseParams,
    PostProcessMaterial,
    PostProcessParams,
)

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.utilities.transformation import Transformation

CameraIntrinsic = pd_sensor_pb2.CameraIntrinsic
LidarIntrinsic = pd_sensor_pb2.LidarIntrinsic
RadarIntrinsic = pd_sensor_pb2.RadarIntrinsic
SensorExtrinsic = pd_sensor_pb2.SensorExtrinsic

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


@register_wrapper(proto_type=pd_sensor_pb2_base.SensorConfig)
class SensorConfig(pd_sensor_pb2.SensorConfig):
    @property
    def name(self) -> str:
        return self.display_name

    @property
    def sensor_to_ego(self) -> Transformation:
        return self.get_sensor_to_ego(in_sim_coordinate_system=False)

    def get_sensor_to_ego(self, in_sim_coordinate_system: bool = False) -> Transformation:
        extrinsic: SensorExtrinsic = self.sensor_extrinsic
        pose = Transformation.from_euler_angles(
            angles=[extrinsic.proto.pitch, extrinsic.proto.roll, extrinsic.proto.yaw],
            order="xyz",
            translation=[extrinsic.proto.x, extrinsic.proto.y, extrinsic.proto.z],
            degrees=True,
        )
        # if not in_sim_coordinate_system: TODO: LFU to FLU (left to right hand problem)
        #     pose = pose @ SIM_TO_INTERNAL_COORDINATE_SYSYEM
        return pose

    @property
    def ego_to_sensor(self) -> Transformation:
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
        sim_pose = self.get_sensor_to_ego(in_sim_coordinate_system=True)
        sim_pose = pd.state.Pose6D.from_transformation_matrix(matrix=sim_pose.transformation_matrix)
        if self.is_camera:
            return CameraSensor(
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
                    DistortionParams, message=self.camera_intrinsic.proto, attr_name="distortion_params"
                ),
                noise_params=convert_to_step_class(
                    NoiseParams, message=self.camera_intrinsic.proto, attr_name="noise_params"
                ),
                post_process_params=convert_to_step_class(
                    PostProcessParams, message=self.camera_intrinsic.proto, attr_name="post_process_params"
                ),
                # post_process_materials=convert_to_step_class(
                #     PostProcessMaterial, message=self.camera_intrinsic.proto, attr_name="post_process"
                # ),
                transmit_gray=self.camera_intrinsic.transmit_gray,
                fisheye_model=self.camera_intrinsic.distortion_params.fisheye_model,
                distortion_lookup_table=self.camera_intrinsic.distortion_lookup_table,
                time_offset=self.camera_intrinsic.time_offset,
            )
        elif self.is_lidar:
            return LiDARSensor(
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
        **kwargs,
    ) -> "SensorConfig":
        if isinstance(pose, np.ndarray):
            pose = Transformation.from_transformation_matrix(pose)

        x, y, z = pose.translation
        pitch, roll, yaw = pose.as_euler_angles(order="xyz", degrees=True)
        extrinsic = SensorExtrinsic(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, follow_rotation=follow_rotation, lock_to_yaw=lock_to_yaw
        )
        annotation_types = annotation_types if annotation_types is not None else []
        return SensorConfig(
            display_name=name,
            # pose=pose,
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


@register_wrapper(proto_type=pd_sensor_pb2_base.SensorRigConfig)
class SensorRig(pd_sensor_pb2.SensorRigConfig):
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
        field_of_view_degrees: float,
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
