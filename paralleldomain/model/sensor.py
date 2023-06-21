from datetime import datetime
from typing import Any, Dict, Generator, Generic, List, Optional, Set, Type, TypeVar, Union

import numpy as np

from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import DecoderImage, Image, ImageDecoderProtocol
from paralleldomain.model.point_cloud import DecoderPointCloud, PointCloud, PointCloudDecoderProtocol
from paralleldomain.model.radar_point_cloud import (
    DecoderRadarPointCloud,
    DecoderRangeDopplerMap,
    RadarPointCloud,
    RadarPointCloudDecoderProtocol,
    RangeDopplerMap,
    RadarFrameHeader,
    DecoderRadarFrameHeader,
    RadarFrameHeaderDecoderProtocol,
)
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup, DistortionLookupTable, project_points_3d_to_2d

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.constants import (
    CAMERA_MODEL_OPENCV_FISHEYE,
    CAMERA_MODEL_OPENCV_PINHOLE,
    CAMERA_MODEL_PD_FISHEYE,
    CAMERA_MODEL_PD_ORTHOGRAPHIC,
)
from paralleldomain.model.annotation import Annotation, AnnotationType
from paralleldomain.model.annotation.albedo_2d import Albedo2D
from paralleldomain.model.annotation.backward_optical_flow import BackwardOpticalFlow
from paralleldomain.model.annotation.backward_scene_flow import BackwardSceneFlow
from paralleldomain.model.annotation.bounding_box_2d import BoundingBoxes2D
from paralleldomain.model.annotation.bounding_box_3d import BoundingBoxes3D
from paralleldomain.model.annotation.depth import Depth
from paralleldomain.model.annotation.instance_segmentation_2d import InstanceSegmentation2D
from paralleldomain.model.annotation.instance_segmentation_3d import InstanceSegmentation3D
from paralleldomain.model.annotation.material_properties_2d import MaterialProperties2D
from paralleldomain.model.annotation.material_properties_3d import MaterialProperties3D
from paralleldomain.model.annotation.optical_flow import OpticalFlow
from paralleldomain.model.annotation.point_2d import Points2D
from paralleldomain.model.annotation.point_cache import PointCaches
from paralleldomain.model.annotation.polygon_2d import Polygons2D
from paralleldomain.model.annotation.polyline_2d import Polylines2D
from paralleldomain.model.annotation.scene_flow import SceneFlow
from paralleldomain.model.annotation.semantic_segmentation_2d import SemanticSegmentation2D
from paralleldomain.model.annotation.semantic_segmentation_3d import SemanticSegmentation3D
from paralleldomain.model.annotation.surface_normals_2d import SurfaceNormals2D
from paralleldomain.model.annotation.surface_normals_3d import SurfaceNormals3D
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")
F = TypeVar("F", Image, PointCloud, Annotation)
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])
SensorDataTypes = Union[
    Type[Image],
    Type[PointCloud],
    Type[BoundingBoxes2D],
    Type[BoundingBoxes3D],
    Type[SemanticSegmentation2D],
    Type[InstanceSegmentation2D],
    Type[SemanticSegmentation3D],
    Type[InstanceSegmentation3D],
    Type[OpticalFlow],
    Type[BackwardOpticalFlow],
    Type[Depth],
    Type[SurfaceNormals3D],
    Type[SurfaceNormals2D],
    Type[SceneFlow],
    Type[BackwardSceneFlow],
    Type[MaterialProperties2D],
    Type[MaterialProperties3D],
    Type[Albedo2D],
    Type[Points2D],
    Type[Polygons2D],
    Type[Polylines2D],
    Type[PointCaches],
]


class FilePathedDataType:
    """Allows to get type-safe access to data types that may have a file linked to them,
    e.g., annotation data, images and point clouds.

    Attributes:
        BoundingBoxes2D
        BoundingBoxes3D
        SemanticSegmentation2D
        InstanceSegmentation2D
        SemanticSegmentation3D
        InstanceSegmentation3D
        OpticalFlow
        BackwardOpticalFlow
        Depth
        SurfaceNormals3D
        SurfaceNormals2D
        SceneFlow
        BackwardSceneFlow
        MaterialProperties2D
        MaterialProperties3D
        Albedo2D
        Points2D
        Polygons2D
        Polylines2D
        Image
        PointCloud

    Examples:
        Access 2D Bounding Box annotation file path for a camera frame:
        ::

            camera_frame: SensorFrame = ...  # get any camera's SensorFrame

            from paralleldomain.model.sensor import FilePathedDataType

            image_file_path = camera_frame.get_file_path(FilePathedDataType.Image)
            if image_file_path is not None:
                with image_file_path.open("r"):
                    # do something
    """

    Image: Type[Image] = Image  # noqa: F811
    PointCloud: Type[PointCloud] = PointCloud  # noqa: F811
    BoundingBoxes2D: Type[BoundingBoxes2D] = BoundingBoxes2D  # noqa: F811
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D  # noqa: F811
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D  # noqa: F811
    InstanceSegmentation2D: Type[InstanceSegmentation2D] = InstanceSegmentation2D  # noqa: F811
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D  # noqa: F811
    InstanceSegmentation3D: Type[InstanceSegmentation3D] = InstanceSegmentation3D  # noqa: F811
    OpticalFlow: Type[OpticalFlow] = OpticalFlow  # noqa: F811
    BackwardOpticalFlow: Type[BackwardOpticalFlow] = BackwardOpticalFlow  # noqa: F811
    Depth: Type[Depth] = Depth  # noqa: F811
    SurfaceNormals3D: Type[SurfaceNormals3D] = SurfaceNormals3D  # noqa: F811
    SurfaceNormals2D: Type[SurfaceNormals2D] = SurfaceNormals2D  # noqa: F811
    SceneFlow: Type[SceneFlow] = SceneFlow  # noqa: F811
    BackwardSceneFlow: Type[BackwardSceneFlow] = BackwardSceneFlow  # noqa: F811
    MaterialProperties2D: Type[MaterialProperties2D] = MaterialProperties2D  # noqa: F811
    MaterialProperties3D: Type[MaterialProperties3D] = MaterialProperties3D  # noqa: F811
    Albedo2D: Type[Albedo2D] = Albedo2D  # noqa: F811
    Points2D: Type[Points2D] = Points2D  # noqa: F811
    Polygons2D: Type[Polygons2D] = Polygons2D  # noqa: F811
    Polylines2D: Type[Polylines2D] = Polylines2D  # noqa: F811
    PointCaches: Type[PointCaches] = PointCaches  # noqa: F811


class CameraModel:
    """Camera Model short hands for value-safe access.

    Attributes:
        OPENCV_PINHOLE: Returns internally used string-representation for OpenCV Pinhole camera model.

            Accepts distortion parameters `(k1,k2,p1,p2[,k3[,k4,k5,k6]])` and uses projection (+ distortion) function
            as described in the
            `OpenCV Pinhole documentation <https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html>`_
        OPENCV_FISHEYE: Returns internally used string-representation for OpenCV Fisheye camera model

            Accepts distortion parameters `(k1,k2,k3,k4)` and uses projection (+ distortion) function
            as described in the
            `OpenCV Fisheye documentation <https://docs.opencv.org/4.5.3/db/d58/group__calib3d__fisheye.html>`_
        PD_FISHEYE: Returns internally used string-representation for Parallel Domain Fisheye camera model

            Uses custom distortion lookup table for translation between non-distorted and distorted angles.

        PD_ORTHOGRAPHIC: Returns internally used string-representation for Parallel Domain Orthographic camera model

            Uses `fx,fy` for pixel per meter resolution. `p1,p2` are used for near- and far-clip plane values.
    """

    OPENCV_PINHOLE: str = CAMERA_MODEL_OPENCV_PINHOLE
    OPENCV_FISHEYE: str = CAMERA_MODEL_OPENCV_FISHEYE
    PD_FISHEYE: str = CAMERA_MODEL_PD_FISHEYE
    PD_ORTHOGRAPHIC: str = CAMERA_MODEL_PD_ORTHOGRAPHIC


class SensorFrameDecoderProtocol(Protocol[TDateTime]):
    @property
    def dataset_name(self) -> str:
        pass

    @property
    def scene_name(self) -> SceneName:
        pass

    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorExtrinsic":
        pass

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorPose":
        pass

    def get_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> List[T]:
        pass

    def get_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        pass

    def get_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass

    def get_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        pass

    def get_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        pass

    def get_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        pass


class SensorFrame(Generic[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: SensorFrameDecoderProtocol[TDateTime],
    ):
        self._frame_id = frame_id
        self._decoder = decoder
        self._sensor_name = sensor_name

    @property
    def dataset_name(self) -> str:
        return self._decoder.dataset_name

    @property
    def scene_name(self) -> SceneName:
        return self._decoder.scene_name

    @property
    def extrinsic(self) -> "SensorExtrinsic":
        """
        Local Sensor coordinate system to vehicle coordinate system
        """
        return self._decoder.get_extrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def pose(self) -> "SensorPose":
        """
        Local Vehicle coordinate system at the current time step to world coordinate system
        """
        return self._decoder.get_sensor_pose(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def ego_to_world(self) -> Transformation:
        """
        Transformation from ego to world coordinate system
        """
        return self.pose

    @property
    def world_to_ego(self) -> Transformation:
        """
        Transformation from world to ego coordinate system
        """
        return self.pose.inverse

    @property
    def ego_to_sensor(self) -> Transformation:
        """
        Transformation from ego to sensor coordinate system
        """
        return self.extrinsic.inverse

    @property
    def sensor_to_ego(self) -> Transformation:
        """
        Transformation from sensor to ego coordinate system
        """
        return self.extrinsic

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        return list(self._annotation_type_identifiers.keys())

    @property
    def class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return self._decoder.get_class_maps()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_metadata(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def _annotation_type_identifiers(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        return self._decoder.get_available_annotation_types(sensor_name=self.sensor_name, frame_id=self.frame_id)

    def get_annotations(self, annotation_type: Type[T]) -> T:
        if annotation_type not in self._annotation_type_identifiers:
            raise ValueError(f"The annotation type {annotation_type} is not available in this sensor frame!")
        return self._decoder.get_annotations(
            sensor_name=self.sensor_name,
            frame_id=self.frame_id,
            identifier=self._annotation_type_identifiers[annotation_type],
            annotation_type=annotation_type,
        )

    def get_file_path(self, data_type: Type[F]) -> Optional[AnyPath]:
        return self._decoder.get_file_path(
            sensor_name=self.sensor_name,
            frame_id=self.frame_id,
            data_type=data_type,
        )

    @property
    def date_time(self) -> TDateTime:
        return self._decoder.get_date_time(sensor_name=self.sensor_name, frame_id=self.frame_id)

    def __lt__(self, other: "SensorFrame[TDateTime]"):
        if self.date_time is not None and other.date_time is not None:
            # if isinstance(other, type(self)):
            return self.date_time < other.date_time
        return id(self) < id(other)


class LidarSensorFrameDecoderProtocol(SensorFrameDecoderProtocol[TDateTime], PointCloudDecoderProtocol):
    ...


class RadarSensorFrameDecoderProtocol(
    SensorFrameDecoderProtocol[TDateTime], RadarPointCloudDecoderProtocol, RadarFrameHeaderDecoderProtocol
):
    ...


class LidarSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: LidarSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def point_cloud(self) -> PointCloud:
        return DecoderPointCloud(decoder=self._decoder, sensor_name=self.sensor_name, frame_id=self.frame_id)


class RadarSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: RadarSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def radar_point_cloud(self) -> RadarPointCloud:
        return DecoderRadarPointCloud(decoder=self._decoder, sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def radar_range_doppler_map(self) -> RangeDopplerMap:
        return DecoderRangeDopplerMap(decoder=self._decoder, sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def header(self) -> RadarFrameHeader:
        return DecoderRadarFrameHeader(decoder=self._decoder, sensor_name=self.sensor_name, frame_id=self.frame_id)


class CameraSensorFrameDecoderProtocol(SensorFrameDecoderProtocol[TDateTime], ImageDecoderProtocol):
    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorIntrinsic":
        pass

    def get_distortion_lookup(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[DistortionLookup]:
        pass


class CameraSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: CameraSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def image(self) -> Image:
        return DecoderImage(decoder=self._decoder, frame_id=self.frame_id, sensor_name=self.sensor_name)

    @property
    def intrinsic(self) -> "SensorIntrinsic":
        return self._decoder.get_intrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def distortion_lookup(self) -> Optional[DistortionLookup]:
        return self._decoder.get_distortion_lookup(sensor_name=self.sensor_name, frame_id=self.frame_id)

    def project_points_from_3d(
        self, points_3d: np.ndarray, distortion_lookup: Optional[DistortionLookup] = None
    ) -> np.ndarray:
        if distortion_lookup is None:
            distortion_lookup = self.distortion_lookup

        return project_points_3d_to_2d(
            k_matrix=self.intrinsic.camera_matrix,
            camera_model=self.intrinsic.camera_model,
            distortion_parameters=self.intrinsic.distortion_parameters,
            distortion_lookup=distortion_lookup,
            points_3d=points_3d,
        )


TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)


class SensorDecoderProtocol(Protocol[TSensorFrameType]):
    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> TSensorFrameType:
        pass

    def get_frame_ids(self, sensor_name: SensorName) -> Set[FrameId]:
        pass


class Sensor(Generic[TSensorFrameType]):
    def __init__(
        self,
        sensor_name: SensorName,
        decoder: SensorDecoderProtocol[TSensorFrameType],
    ):
        self._decoder = decoder
        self._sensor_name = sensor_name

    @property
    def name(self) -> str:
        return self._sensor_name

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids(sensor_name=self.name)

    def get_frame(self, frame_id: FrameId) -> TSensorFrameType:
        return self._decoder.get_sensor_frame(frame_id=frame_id, sensor_name=self._sensor_name)

    @property
    def sensor_frames(self) -> Generator[TSensorFrameType, None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class CameraSensor(Sensor[CameraSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> CameraSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id, sensor_name=self._sensor_name)

    @property
    def sensor_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class LidarSensor(Sensor[LidarSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> LidarSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id, sensor_name=self._sensor_name)

    @property
    def sensor_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class RadarSensor(Sensor[RadarSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> RadarSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id, sensor_name=self._sensor_name)


class SensorPose(Transformation):
    ...


class SensorExtrinsic(Transformation):
    ...


class SensorIntrinsic:
    def __init__(
        self,
        cx=0.0,
        cy=0.0,
        fx=0.0,
        fy=0.0,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
        k4=0.0,
        k5=0.0,
        k6=0.0,
        skew=0.0,
        fov=0.0,
        camera_model=CameraModel.OPENCV_PINHOLE,
    ):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.skew = skew
        self.fov = fov
        self.camera_model = camera_model

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, self.skew, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ]
        )

    @property
    def distortion_parameters(self) -> Optional[np.ndarray]:
        if self.camera_model == CAMERA_MODEL_OPENCV_PINHOLE:
            return np.array(
                [
                    self.k1,
                    self.k2,
                    self.p1,
                    self.p2,
                    self.k3,
                    self.k4,
                    self.k5,
                    self.k6,
                ]
            )
        elif self.camera_model == CAMERA_MODEL_OPENCV_FISHEYE:
            return np.array(
                [
                    self.k1,
                    self.k2,
                    self.k3,
                    self.k4,
                ]
            )
        elif self.camera_model == CAMERA_MODEL_PD_FISHEYE:
            return None
        elif self.camera_model == CAMERA_MODEL_PD_ORTHOGRAPHIC:
            return None
        else:
            raise NotImplementedError(f"No distortion parameters implemented for camera model {self.camera_model}")

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        if isinstance(other, np.ndarray):
            ori_shape = other.shape
            needs_transpose = False
            if len(other.shape) != 2:
                ori_shape = other.shape
                if ori_shape[0] == 3:
                    reshaped_other = other.reshape(3, -1)
                elif ori_shape[-1] == 3:
                    needs_transpose = True
                    reshaped_other = other.reshape(-1, 3).T
                else:
                    raise ValueError(f"unsupported shape {ori_shape}! First or last dim has to be 3!")
            else:
                reshaped_other = other

            projected = self.camera_matrix @ reshaped_other
            if needs_transpose:
                projected = projected.T
            return projected.reshape(ori_shape)
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4xn numpy array!")
