from __future__ import annotations

import math
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np

from paralleldomain.constants import (
    CAMERA_MODEL_OPENCV_FISHEYE,
    CAMERA_MODEL_OPENCV_PINHOLE,
    CAMERA_MODEL_PD_FISHEYE,
    CAMERA_MODEL_PD_ORTHOGRAPHIC,
)
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import DecoderImage, Image, ImageDecoderProtocol
from paralleldomain.model.point_cloud import DecoderPointCloud, PointCloud, PointCloudDecoderProtocol
from paralleldomain.model.radar_point_cloud import (
    DecoderRadarFrameHeader,
    DecoderRadarPointCloud,
    DecoderRangeDopplerMap,
    RadarFrameHeader,
    RadarFrameHeaderDecoderProtocol,
    RadarPointCloud,
    RadarPointCloudDecoderProtocol,
    RangeDopplerMap,
)
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup, fov_to_focal_length, project_points_3d_to_2d
from paralleldomain.utilities.transformation import Transformation

if TYPE_CHECKING:
    from paralleldomain.model.frame import Frame
    from paralleldomain.model.scene import Scene
    from paralleldomain.model.unordered_scene import UnorderedScene


SelfType = TypeVar("SelfType", bound="SensorFrame")
T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])
SensorDataCopyTypes = Union[
    Type[Image],
    Type[PointCloud],
    AnnotationIdentifier,
]


class FilePathedDataType(AnnotationTypes):
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
    RadarPointCloud: Type[RadarPointCloud] = RadarPointCloud  # noqa: F811
    RadarFrameHeader: Type[RadarFrameHeader] = RadarFrameHeader  # noqa: F811
    RangeDopplerMap: Type[RangeDopplerMap] = RangeDopplerMap  # noqa: F811


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

    @property
    def sensor_name(self) -> SensorName:
        pass

    @property
    def frame_id(self) -> FrameId:
        pass

    def get_extrinsic(self) -> "SensorExtrinsic":
        pass

    def get_sensor_pose(self) -> "SensorPose":
        pass

    def get_annotations(self, identifier: AnnotationIdentifier[T]) -> List[T]:
        pass

    def get_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        pass

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        pass

    def get_date_time(self) -> TDateTime:
        pass

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        pass

    def get_scene(self) -> Union[Scene, UnorderedScene]:
        pass


class SensorFrame(Generic[TDateTime]):
    def __init__(
        self,
        decoder: SensorFrameDecoderProtocol[TDateTime],
    ):
        self._decoder = decoder

    @property
    def dataset_name(self) -> str:
        return self._decoder.dataset_name

    @property
    def scene_name(self) -> SceneName:
        return self._decoder.scene_name

    @property
    def scene(self) -> Union[Scene, UnorderedScene]:
        return self._decoder.get_scene()

    @property
    def sensor(self) -> "Sensor":
        return self.scene.get_sensor(sensor_name=self.sensor_name)

    @property
    def frame(self) -> Frame:
        return self.scene.get_frame(frame_id=self.frame_id)

    @property
    def next_frame(self) -> Optional[Frame[TDateTime]]:
        """
        Returns: The next frame in the dataset. None if there is no next frame
        or the scene has no order in frame ids. Note that that frame might not have data for this sensor.
        """
        return self.frame.next_frame

    @property
    def previous_frame(self) -> Optional[Frame[TDateTime]]:
        """
        Returns: The previous frame in the dataset. None if there is no previous frame
        or the scene has no order in frame ids. Note that that frame might not have data for this sensor.
        """
        return self.frame.previous_frame

    @property
    def next_sensor_frame_id(self) -> Optional[FrameId]:
        """
        Returns: The frame id of the next frame that has data for this sensor. None if there is no next frame
        or the scene has no order in frame ids.
        """
        scene = self.scene
        if not scene.is_ordered:
            return None
        sensor_fids = scene.get_sensor(sensor_name=self.sensor_name).frame_ids
        frame_ids = [fid for fid in scene.frame_ids if fid in sensor_fids]
        next_frame_id_idx = frame_ids.index(self.frame_id) + 1
        if 0 <= next_frame_id_idx < len(frame_ids):
            return frame_ids[next_frame_id_idx]
        return None

    @property
    def previous_sensor_frame_id(self) -> Optional[FrameId]:
        """
        Returns: The frame id of the previous frame that has data for this sensor. None if there is no previous frame
        or the scene has no order in frame ids.
        """
        scene = self.scene
        if not scene.is_ordered:
            return None
        sensor_fids = scene.get_sensor(sensor_name=self.sensor_name).frame_ids
        frame_ids = [fid for fid in scene.frame_ids if fid in sensor_fids]
        previous_frame_id_idx = frame_ids.index(self.frame_id) - 1
        if 0 <= previous_frame_id_idx < len(frame_ids):
            return frame_ids[previous_frame_id_idx]
        return None

    @property
    def next_frame_id(self) -> Optional[FrameId]:
        """
        Returns: The frame id of the next frame in the dataset. None if there is no next frame
        or the scene has no order in frame ids. Note that that frame might not have data for this sensor.
        """
        return self.frame.next_frame_id

    @property
    def previous_frame_id(self) -> Optional[FrameId]:
        """
        Returns: The frame id of the previous frame in the dataset. None if there is no previous frame
        or the scene has no order in frame ids. Note that that frame might not have data for this sensor.
        """
        return self.frame.previous_frame_id

    @property
    def next_sensor_frame(self: SelfType) -> Optional[SelfType]:
        next_sensor_frame_id = self.next_sensor_frame_id
        if next_sensor_frame_id is not None:
            next_frame = self.scene.get_frame(frame_id=next_sensor_frame_id)
            return next_frame.get_sensor(sensor_name=self.sensor_name)
        return None

    @property
    def previous_sensor_frame(self: SelfType) -> Optional[SelfType]:
        previous_sensor_frame_id = self.previous_sensor_frame_id
        if previous_sensor_frame_id is not None:
            previous_frame = self.scene.get_frame(frame_id=previous_sensor_frame_id)
            return previous_frame.get_sensor(sensor_name=self.sensor_name)
        return None

    def _get_extrinsic(self) -> "SensorExtrinsic":
        """
        Local Sensor coordinate system to vehicle coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. Vehicle coordinate system is in FLU.
        """
        return self._decoder.get_extrinsic()

    @property
    def extrinsic(self) -> "SensorExtrinsic":
        """
        Local Sensor coordinate system to vehicle coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. Vehicle coordinate system is in FLU.
        """
        return self._get_extrinsic()

    def _get_pose(self) -> "SensorPose":
        """
        Local Sensor coordinate system at the current time step to world coordinate system.
        This is the same as the sensor_to_world property. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. World coordinate system is Z up.
        """
        return self._decoder.get_sensor_pose()

    @property
    def pose(self) -> "SensorPose":
        """
        Local Sensor coordinate system at the current time step to world coordinate system.
        This is the same as the sensor_to_world property. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. World coordinate system is Z up.
        """
        return self._get_pose()

    @property
    def sensor_to_world(self) -> Transformation:
        """
        Transformation from ego to world coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. World coordinate system is Z up.
        """
        return self.pose

    @property
    def world_to_sensor(self) -> Transformation:
        """
        Transformation from world to ego coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. World coordinate system is Z up.
        """
        return self.pose.inverse

    @property
    def ego_to_world(self) -> Transformation:
        """
        Transformation from ego to world coordinate system
        """
        return self.sensor_to_world @ self.ego_to_sensor

    @property
    def world_to_ego(self) -> Transformation:
        """
        Transformation from world to ego coordinate system
        """
        return self.ego_to_world.inverse

    @property
    def ego_to_sensor(self) -> Transformation:
        """
        Transformation from ego to sensor coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. Vehicle coordinate system is in FLU.
        """
        return self.extrinsic.inverse

    @property
    def sensor_to_ego(self) -> Transformation:
        """
        Transformation from sensor to ego coordinate system. Local Sensor coordinates are in RDF for PD data,
        but it can be something else for other dataset types. Vehicle coordinate system is in FLU.
        """
        return self.extrinsic

    @property
    def sensor_name(self) -> str:
        return self._decoder.sensor_name

    @property
    def frame_id(self) -> FrameId:
        return self._decoder.frame_id

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        return list({ai.annotation_type for ai in self.available_annotation_identifiers})

    def _get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._decoder.get_available_annotation_identifiers()

    @property
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._get_available_annotation_identifiers()

    def get_annotation_identifiers_of_type(self, annotation_type: Type[T]) -> List[AnnotationIdentifier[T]]:
        return [
            identifier
            for identifier in self.available_annotation_identifiers
            if issubclass(identifier.annotation_type, annotation_type)
        ]

    def _get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self._decoder.get_class_maps()

    @property
    def class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self._get_class_maps()

    @overload
    def get_class_map(self, annotation_type: Type[T], name: str = None) -> ClassMap:
        pass

    @overload
    def get_class_map(self, annotation_identifier: AnnotationIdentifier[T]) -> ClassMap:
        pass

    def get_class_map(
        self, annotation_type: Type[T] = None, annotation_identifier: AnnotationIdentifier[T] = None, name: str = None
    ) -> ClassMap:
        annotation_identifier = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=self.available_annotation_identifiers,
            annotation_type=annotation_type,
            annotation_identifier=annotation_identifier,
            name=name,
        )
        return self.class_maps[annotation_identifier]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_metadata()

    def get_annotation_identifiers(
        self, names: Optional[List[str]] = None, annotation_types: Optional[List[AnnotationType]] = None
    ) -> List[AnnotationIdentifier]:
        return [
            ai
            for ai in self.available_annotation_identifiers
            if (names is None or ai.name in names)
            and (annotation_types is None or ai.annotation_type in annotation_types)
        ]

    @overload
    def get_annotations(self, annotation_identifier: AnnotationIdentifier[T]) -> T:
        ...

    @overload
    def get_annotations(self, annotation_type: Type[T], name: str = None) -> T:
        ...

    def get_annotations(
        self, annotation_type: Type[T] = None, annotation_identifier: AnnotationIdentifier[T] = None, name: str = None
    ) -> T:
        annotation_identifier = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=self.available_annotation_identifiers,
            annotation_type=annotation_type,
            annotation_identifier=annotation_identifier,
            name=name,
        )
        return self._decoder.get_annotations(
            identifier=annotation_identifier,
        )

    def get_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        # Note: We also support Type[Annotation] for data_type for backwards compatibility
        return self._decoder.get_file_path(
            data_type=data_type,
        )

    @property
    def date_time(self) -> TDateTime:
        return self._decoder.get_date_time()

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
        decoder: LidarSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(decoder=decoder)
        self._decoder = decoder

    def _get_point_cloud(self) -> PointCloud:
        return DecoderPointCloud(decoder=self._decoder)

    @property
    def point_cloud(self) -> PointCloud:
        return self._get_point_cloud()


class RadarSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        decoder: RadarSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(decoder=decoder)
        self._decoder = decoder

    def _get_radar_point_cloud(self) -> RadarPointCloud:
        return DecoderRadarPointCloud(decoder=self._decoder)

    def _get_radar_range_doppler_map(self) -> RangeDopplerMap:
        return DecoderRangeDopplerMap(decoder=self._decoder)

    def _get_header(self) -> RadarFrameHeader:
        return DecoderRadarFrameHeader(decoder=self._decoder)

    @property
    def radar_point_cloud(self) -> RadarPointCloud:
        return self._get_radar_point_cloud()

    @property
    def radar_range_doppler_map(self) -> RangeDopplerMap:
        return self._get_radar_range_doppler_map()

    @property
    def header(self) -> RadarFrameHeader:
        return self._get_header()


class CameraSensorFrameDecoderProtocol(SensorFrameDecoderProtocol[TDateTime], ImageDecoderProtocol):
    def get_intrinsic(self) -> "SensorIntrinsic":
        pass

    def get_distortion_lookup(self) -> Optional[DistortionLookup]:
        pass


class CameraSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        decoder: CameraSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(decoder=decoder)
        self._decoder = decoder

    def _get_image(self) -> Image:
        return DecoderImage(decoder=self._decoder)

    @property
    def image(self) -> Image:
        return self._get_image()

    def _get_intrinsic(self) -> "SensorIntrinsic":
        return self._decoder.get_intrinsic()

    @property
    def intrinsic(self) -> "SensorIntrinsic":
        return self._get_intrinsic()

    def _get_distortion_lookup(self) -> Optional[DistortionLookup]:
        return self._decoder.get_distortion_lookup()

    @property
    def distortion_lookup(self) -> Optional[DistortionLookup]:
        return self._get_distortion_lookup()

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
    def get_sensor_frame(self, frame_id: FrameId) -> TSensorFrameType:
        pass

    def get_frame_ids(self) -> Set[FrameId]:
        pass

    def get_scene(self) -> Union[Scene, UnorderedScene]:
        pass

    @property
    def dataset_name(self) -> str:
        pass

    @property
    def scene_name(self) -> SceneName:
        pass

    @property
    def sensor_name(self) -> SensorName:
        pass


class Sensor(Generic[TSensorFrameType]):
    def __init__(
        self,
        decoder: SensorDecoderProtocol[TSensorFrameType],
    ):
        self._decoder = decoder

    @property
    def name(self) -> str:
        return self._decoder.sensor_name

    def _get_frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids()

    @property
    def scene(self) -> Union[Scene, UnorderedScene]:
        return self._decoder.get_scene()

    @property
    def dataset_name(self) -> str:
        return self._decoder.dataset_name

    @property
    def scene_name(self) -> SceneName:
        return self._decoder.scene_name

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._get_frame_ids()

    def get_frame(self, frame_id: FrameId) -> TSensorFrameType:
        return self._decoder.get_sensor_frame(frame_id=frame_id)

    @property
    def sensor_frames(self) -> Generator[TSensorFrameType, None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class CameraSensor(Sensor[CameraSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> CameraSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id)

    @property
    def sensor_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class LidarSensor(Sensor[LidarSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> LidarSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id)

    @property
    def sensor_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids)


class RadarSensor(Sensor[RadarSensorFrame[TDateTime]]):
    def get_frame(self, frame_id: FrameId) -> RadarSensorFrame[TDateTime]:
        return self._decoder.get_sensor_frame(frame_id=frame_id)


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

    @staticmethod
    def from_field_of_view(field_of_view_degrees: float, width: int, height: int) -> SensorIntrinsic:
        fx = fy = fov_to_focal_length(
            fov=math.radians(field_of_view_degrees), sensor_size=width if width >= height else height
        )
        return SensorIntrinsic(
            cx=width / 2,
            cy=height / 2,
            fx=fx,
            fy=fy,
            fov=field_of_view_degrees,
            camera_model=CameraModel.OPENCV_PINHOLE,
        )
