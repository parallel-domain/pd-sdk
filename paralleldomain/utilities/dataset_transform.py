import abc
import functools
from copy import copy
from dataclasses import field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union, overload

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import Annotation, AnnotationIdentifier, AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.frame import Frame
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.radar_point_cloud import RadarFrameHeader, RadarPointCloud, RangeDopplerMap
from paralleldomain.model.sensor import (
    CameraSensorFrame,
    LidarSensorFrame,
    RadarSensorFrame,
    Sensor,
    SensorExtrinsic,
    SensorFrame,
    SensorIntrinsic,
    SensorPose,
)
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.projection import DistortionLookup

ModelTypes = Union[SensorFrame, Scene, UnorderedScene, Dataset, Frame, Sensor]
D = TypeVar(
    "D",
    Annotation,
    Image,
    PointCloud,
    RadarPointCloud,
    RangeDopplerMap,
    RadarFrameHeader,
    ClassMap,
    SensorExtrinsic,
    SensorPose,
    SensorIntrinsic,
    DistortionLookup,
    List[AnnotationIdentifier],
    List[AnnotationType],
    List[SensorName],
    List[SceneName],
    List[FrameId],
)
# T = TypeVar("T", SensorFrame, Scene, UnorderedScene, Dataset, Frame, Sensor)
M = TypeVar("M", SensorFrame, Scene, UnorderedScene, Dataset, Frame, Sensor, ModelTypes)
A = TypeVar("A")
DataObjectTypes = Union[
    Type[Image],
    Type[PointCloud],
    Type[RadarPointCloud],
    Type[ClassMap],
    Type[DistortionLookup],
    Type[SensorIntrinsic],
    Type[SensorPose],
    Type[SensorExtrinsic],
    Type[RadarFrameHeader],
    Type[RangeDopplerMap],
    Type[RadarPointCloud],
    Type[PointCloud],
    Type[Annotation],
    AnnotationIdentifier,
]


class DataTransformer(Generic[D, M]):
    @abc.abstractmethod
    def __call__(self, model: M) -> D:
        pass


class DatasetTransformation:
    """
    This class can be used to filter and transform / alter a dataset. It lets you filter sensors, frames, add new
    annotation types, change annotations etc. This is useful if you want to plug a dataset directly into a
    framework like torch or tensorflow for training or re encode it into a different format to support your
    training pipeline. You can find some examples at examples/dataset_transformation
    """

    def __init__(
        self,
        transformations: Dict[DataObjectTypes, DataTransformer] = None,
        annotation_identifiers: Optional[
            Union[List[Union[AnnotationIdentifier, AnnotationType]], DataTransformer]
        ] = None,
        frame_ids: Optional[Union[List[FrameId], DataTransformer]] = None,
        sensor_names: Optional[Union[List[SensorName], DataTransformer]] = None,
        scene_names: Optional[Union[List[SceneName], DataTransformer]] = None,
    ):
        """

        Args:
            transformations: a Dict that maps from a data type to the transformer that will provide that type
            annotation_identifiers: A list of annotation identifiers to filter the dataset by. Also takes a
            transformer that can calculate the available annotations based on a scene or
            sensor frame that will be passed to it.
            frame_ids: A list of frame ids to filter the dataset by. Also takes a
            transformer that can calculate the available frame ids based on a scene or sensor that will be passed
            to it.
            sensor_names: A list of sensor names to filter the dataset by. Also takes a
            transformer that can calculate the available sensor names based on a scene or frame that will be passed
            to it.
            scene_names: A list of scene names to filter the dataset by. Also takes a
            transformer that can calculate the available scene names based on the dataset that will be passed to it.
        """
        self.transformations = transformations if transformations is not None else dict()
        self.annotation_identifiers = annotation_identifiers
        self.frame_ids = frame_ids
        self.sensor_names = sensor_names
        self.scene_names = scene_names

        self._identifier_transformation = {
            k: v for k, v in self.transformations.items() if isinstance(k, AnnotationIdentifier)
        }
        self._annotation_type_transformations = {
            k: v for k, v in self.transformations.items() if isinstance(k, type) and issubclass(k, Annotation)
        }

        if annotation_identifiers is not None:
            self._available_identifiers = (
                [k for k in self.annotation_identifiers if isinstance(k, AnnotationIdentifier)]
                if isinstance(annotation_identifiers, list)
                else list()
            )

            self._available_annotation_types = (
                [k for k in self.annotation_identifiers if isinstance(k, type) and issubclass(k, Annotation)]
                if isinstance(annotation_identifiers, list)
                else list()
            )

    @staticmethod
    def _set_original_model(model: ModelTypes):
        if not hasattr(model, "_original_model"):
            model._original_model = copy(model)

    def _wrap_annotations(self, sensor_frame: SensorFrame):
        if (
            self.transformations is None
            or len(self.transformations) == 0
            or not any(
                [isinstance(k, AnnotationIdentifier) or issubclass(k, Annotation) for k in self.transformations.keys()]
            )
        ):
            return

        self._set_original_model(model=sensor_frame)

        @functools.wraps(sensor_frame.get_annotations)
        def wrapper(*args, **kwargs):
            if "annotation_identifier" in kwargs or "annotation_type" in kwargs:
                if "annotation_type" in kwargs:
                    annotation_identifier = AnnotationIdentifier(
                        annotation_type=kwargs["annotation_type"], name=kwargs.get("name", None)
                    )
                else:
                    annotation_identifier = kwargs["annotation_identifier"]

                if annotation_identifier in self._identifier_transformation:
                    val = self._identifier_transformation[annotation_identifier](model=sensor_frame._original_model)
                elif annotation_identifier.annotation_type in self._annotation_type_transformations:
                    val = self._annotation_type_transformations[annotation_identifier.annotation_type](
                        model=sensor_frame._original_model
                    )
                else:
                    val = sensor_frame._original_model.get_annotations(*args, **kwargs)
            return val

        sensor_frame.get_annotations = wrapper

    def _wrap_dataset_get_metadata(self, dataset: Dataset):
        func_ref = dataset._get_metadata
        if self.annotation_identifiers is None:
            return

        self._set_original_model(model=dataset)

        @functools.wraps(dataset._get_metadata)
        def wrapper(*args, **kwargs):
            val = func_ref()

            if isinstance(self.annotation_identifiers, DataTransformer):
                available_annotation_identifiers = self.annotation_identifiers(model=dataset)
            else:
                available_annotation_identifiers = [
                    v
                    for v in val.available_annotation_identifiers
                    if v in self._available_identifiers or v.annotation_type in self._available_annotation_types
                ]

            val = DatasetMeta(
                name=val.name,
                custom_attributes=val.custom_attributes,
                available_annotation_identifiers=available_annotation_identifiers,
            )
            return val

        dataset._get_metadata = wrapper

    def _wrap_available_annotations_call(
        self,
        model: ModelTypes,
        func_ref: Callable[[Any], List[AnnotationIdentifier]],
    ) -> Callable:
        if self.annotation_identifiers is None:
            return

        DatasetTransformation._set_original_model(model=model)

        @functools.wraps(func_ref)
        def wrapper(*args, **kwargs):
            if isinstance(self.annotation_identifiers, DataTransformer):
                val = self.annotation_identifiers(model=model._original_model)
            elif isinstance(self.annotation_identifiers, list):
                val: List[AnnotationIdentifier] = getattr(model._original_model, func_ref.__name__)(*args, **kwargs)
                val = [
                    v
                    for v in val
                    if v in self._available_identifiers or v.annotation_type in self._available_annotation_types
                ]
            return val

        setattr(model, func_ref.__name__, wrapper)

    @staticmethod
    def _wrap_collection_call(
        model: ModelTypes,
        func_ref: Callable,
        filter: Optional[Union[List[AnnotationIdentifier], List[AnnotationType], DataTransformer]],
    ) -> Callable:
        if filter is None:
            return func_ref

        DatasetTransformation._set_original_model(model=model)

        @functools.wraps(func_ref)
        def wrapper(*args, **kwargs):
            if isinstance(filter, DataTransformer):
                val = filter(model=model._original_model)
            else:
                val = getattr(model._original_model, func_ref.__name__)(*args, **kwargs)
                if isinstance(val, list):
                    val = [v for v in val if v in filter]
                elif isinstance(val, set):
                    val = {v for v in val if v in filter}
                elif isinstance(val, dict):
                    val = {k: v for k, v in val.items() if k in filter}
            return val

        setattr(model, func_ref.__name__, wrapper)

    def wrap_methods(self, model: M) -> M:
        if isinstance(model, SensorFrame):
            self._wrap_annotations(sensor_frame=model)
            self._wrap_available_annotations_call(
                model=model,
                func_ref=model._get_available_annotation_identifiers,
            )
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_class_maps,
                filter=self.transformations.get(ClassMap, None),
            )
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_extrinsic,
                filter=self.transformations.get(SensorExtrinsic, None),
            )
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_pose,
                filter=self.transformations.get(SensorPose, None),
            )
            if isinstance(model, CameraSensorFrame):
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_intrinsic,
                    filter=self.transformations.get(SensorIntrinsic, None),
                )
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_distortion_lookup,
                    filter=self.transformations.get(DistortionLookup, None),
                )
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_image,
                    filter=self.transformations.get(Image, None),
                )
            elif isinstance(model, LidarSensorFrame):
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_point_cloud,
                    filter=self.transformations.get(PointCloud, None),
                )
            elif isinstance(model, RadarSensorFrame):
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_radar_point_cloud,
                    filter=self.transformations.get(RadarPointCloud, None),
                )
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_radar_range_doppler_map,
                    filter=self.transformations.get(RangeDopplerMap, None),
                )
                self._wrap_collection_call(
                    model=model,
                    func_ref=model._get_header,
                    filter=self.transformations.get(RadarFrameHeader, None),
                )

        elif isinstance(model, UnorderedScene) or isinstance(model, Scene):
            self._wrap_available_annotations_call(
                model=model,
                func_ref=model._get_available_annotation_identifiers,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_camera_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_lidar_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_radar_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_sensor_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_frame_ids,
                filter=self.frame_ids,
            )
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_class_maps,
                filter=self.transformations.get(ClassMap, None),
            )

        elif isinstance(model, Sensor):
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_frame_ids,
                filter=self.frame_ids,
            )

        elif isinstance(model, Frame):
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_camera_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_lidar_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_radar_names,
                filter=self.sensor_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_sensor_names,
                filter=self.sensor_names,
            )

        elif isinstance(model, Dataset):
            self._wrap_collection_call(
                model=model,
                func_ref=model._get_scene_names,
                filter=self.scene_names,
            )

            self._wrap_collection_call(
                model=model,
                func_ref=model._get_unordered_scene_names,
                filter=self.scene_names,
            )
            self._wrap_dataset_get_metadata(dataset=model)

        return model

    def apply(self, other: M) -> M:
        # we store the transform in the decoder settings to make sure that we can apply this transform to all
        # child models, like frames, sensors etc. Those will be passed to wrap_methods when they are created.
        # See base decoder classes
        other._decoder.settings.model_decorator = self.wrap_methods
        return self.wrap_methods(other)
