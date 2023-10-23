import random
from datetime import datetime
from itertools import chain
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import pypeln

from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import (
    CameraSensor,
    CameraSensorFrame,
    LidarSensor,
    LidarSensorFrame,
    RadarSensor,
    RadarSensorFrame,
    SensorFrame,
    TSensorFrameType,
)
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.generator_shuffle import nested_generator_random_draw

T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class UnorderedSceneDecoderProtocol(Protocol[TDateTime]):
    @property
    def scene_name(self) -> SceneName:
        pass

    @property
    def dataset_name(self) -> str:
        pass

    def get_set_description(self) -> str:
        pass

    def get_set_metadata(self) -> Dict[str, Any]:
        pass

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        pass

    def get_frame(
        self,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        pass

    def get_sensor_names(self) -> List[str]:
        pass

    def get_camera_names(self) -> List[str]:
        pass

    def get_lidar_names(self) -> List[str]:
        pass

    def get_radar_names(self) -> List[str]:
        pass

    def get_frame_ids(self) -> Set[FrameId]:
        pass

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        pass

    def get_camera_sensor(self, camera_name: SensorName) -> CameraSensor[TDateTime]:
        pass

    def get_lidar_sensor(self, lidar_name: SensorName) -> LidarSensor[TDateTime]:
        pass

    def get_radar_sensor(self, radar_name: SensorName) -> RadarSensor[TDateTime]:
        pass

    def clear_from_cache(self):
        pass


class UnorderedScene(Generic[TDateTime]):
    """The sensor frames of a UnorderedScene are not temporally ordered.

    Args:
        name: Name of scene
        decoder: Decoder instance to be used for loading all relevant objects (frames, annotations etc.)
    """

    def __init__(
        self,
        decoder: UnorderedSceneDecoderProtocol[TDateTime],
        is_ordered: bool = False,
    ):
        self.is_ordered = is_ordered
        self._decoder = decoder
        self._number_of_camera_frames = None
        self._number_of_lidar_frames = None
        self._number_of_radar_frames = None
        self._number_of_sensor_frames = None

    @property
    def name(self) -> str:
        return self._decoder.scene_name

    @property
    def dataset_name(self) -> str:
        return self._decoder.dataset_name

    @property
    def description(self) -> str:
        return self._decoder.get_set_description()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_set_metadata()

    @property
    def frames(self) -> Set[Frame[TDateTime]]:
        return {self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids}

    def _get_frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids()

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._get_frame_ids()

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

    def _get_sensor_names(self) -> List[str]:
        return self._decoder.get_sensor_names()

    @property
    def sensor_names(self) -> List[str]:
        return self._get_sensor_names()

    def _get_camera_names(self) -> List[str]:
        return self._decoder.get_camera_names()

    @property
    def camera_names(self) -> List[str]:
        return self._get_camera_names()

    def _get_lidar_names(self) -> List[str]:
        return self._decoder.get_lidar_names()

    @property
    def lidar_names(self) -> List[str]:
        return self._get_lidar_names()

    def _get_radar_names(self) -> List[str]:
        return self._decoder.get_radar_names()

    @property
    def radar_names(self) -> List[str]:
        return self._get_radar_names()

    def get_frame(self, frame_id: FrameId) -> Frame[TDateTime]:
        return self._decoder.get_frame(frame_id=frame_id)

    @property
    def sensors(
        self,
    ) -> Generator[
        Union[
            CameraSensor[CameraSensorFrame[TDateTime]],
            LidarSensor[LidarSensorFrame[TDateTime]],
            RadarSensor[RadarSensorFrame[TDateTime]],
        ],
        None,
        None,
    ]:
        return (a for a in chain(self.cameras, self.lidars, self.radars))

    @property
    def cameras(self) -> Generator[CameraSensor[TSensorFrameType], None, None]:
        yield from self.sensor_pipeline(shuffle=False, concurrent=False, sensor_names=self.camera_names)

    @property
    def lidars(self) -> Generator[LidarSensor[TSensorFrameType], None, None]:
        yield from self.sensor_pipeline(shuffle=False, concurrent=False, sensor_names=self.lidar_names)

    @property
    def radars(self) -> Generator[RadarSensor[TSensorFrameType], None, None]:
        yield from self.sensor_pipeline(shuffle=False, concurrent=False, sensor_names=self.radar_names)

    def get_sensor(
        self, sensor_name: SensorName
    ) -> Union[
        CameraSensor[CameraSensorFrame[TDateTime]],
        LidarSensor[LidarSensorFrame[TDateTime]],
        RadarSensor[RadarSensorFrame[TDateTime]],
    ]:
        if sensor_name in self.camera_names:
            return self.get_camera_sensor(camera_name=sensor_name)
        elif sensor_name in self.lidar_names:
            return self.get_lidar_sensor(lidar_name=sensor_name)
        elif sensor_name in self.radar_names:
            return self.get_radar_sensor(radar_name=sensor_name)
        else:
            raise ValueError(f"Sensor {sensor_name} could not be found.")

    def get_camera_sensor(self, camera_name: SensorName) -> CameraSensor[CameraSensorFrame[TDateTime]]:
        if camera_name not in self.camera_names:
            raise ValueError(f"Camera {camera_name} could not be found.")
        return self._decoder.get_camera_sensor(camera_name=camera_name)

    def get_lidar_sensor(self, lidar_name: SensorName) -> LidarSensor[LidarSensorFrame[TDateTime]]:
        if lidar_name not in self.lidar_names:
            raise ValueError(f"LiDAR {lidar_name} could not be found.")
        return self._decoder.get_lidar_sensor(lidar_name=lidar_name)

    def get_radar_sensor(self, radar_name: SensorName) -> RadarSensor[RadarSensorFrame[TDateTime]]:
        if radar_name not in self.radar_names:
            raise ValueError(f"LiDAR {radar_name} could not be found.")
        return self._decoder.get_radar_sensor(radar_name=radar_name)

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

    def get_sensor_frames(
        self,
        sensor_names: Optional[Iterable[SensorName]] = None,
        frame_ids: Optional[Iterable[FrameId]] = None,
        shuffle: bool = False,
        concurrent: bool = False,
        max_queue_size: int = 8,
        max_workers: int = 4,
        random_seed: int = 42,
        only_cameras: bool = False,
        only_radars: bool = False,
        only_lidars: bool = False,
    ) -> Generator[SensorFrame[TDateTime], None, None]:
        for sensor_frame, _, _ in self.sensor_frame_pipeline(
            shuffle=shuffle,
            concurrent=concurrent,
            sensor_names=sensor_names,
            frame_ids=frame_ids,
            only_cameras=only_cameras,
            only_lidars=only_lidars,
            only_radars=only_radars,
            max_queue_size=max_queue_size,
            max_workers=max_workers,
            random_seed=random_seed,
        ):
            yield sensor_frame

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_cameras=True):
            yield sensor_frame

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_lidars=True):
            yield sensor_frame

    @property
    def radar_frames(self) -> Generator[RadarSensorFrame[TDateTime], None, None]:
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_radars=True):
            yield sensor_frame

    @property
    def sensor_frames(self) -> Generator[SensorFrame[TDateTime], None, None]:
        yield from self.sensor_frame_pipeline(shuffle=False)

    @property
    def number_of_camera_frames(self) -> int:
        if self._number_of_camera_frames is None:
            self._number_of_camera_frames = 0
            for frame in self.frame_pipeline(shuffle=True, concurrent=True):
                self._number_of_camera_frames += len(frame.camera_names)
        return self._number_of_camera_frames

    @property
    def number_of_lidar_frames(self) -> int:
        if self._number_of_lidar_frames is None:
            self._number_of_lidar_frames = 0
            for frame in self.frame_pipeline(shuffle=True, concurrent=True):
                self._number_of_lidar_frames += len(frame.lidar_names)
        return self._number_of_lidar_frames

    @property
    def number_of_radar_frames(self) -> int:
        if self._number_of_radar_frames is None:
            self._number_of_radar_frames = 0
            for frame in self.frame_pipeline(shuffle=True, concurrent=True):
                self._number_of_radar_frames += len(frame.radar_names)
        return self._number_of_radar_frames

    @property
    def number_of_sensor_frames(self) -> int:
        if self._number_of_sensor_frames is None:
            if (
                self._number_of_radar_frames is not None
                and self._number_of_radar_frames is not None
                and self._number_of_radar_frames is not None
            ):
                self._number_of_sensor_frames = (
                    self.number_of_lidar_frames + self.number_of_camera_frames + self.number_of_radar_frames
                )
            else:
                self._number_of_sensor_frames = 0
                for frame in self.frame_pipeline(shuffle=True, concurrent=True):
                    self._number_of_sensor_frames += len(frame.sensor_names)
        return self._number_of_sensor_frames

    def frame_pipeline(
        self,
        shuffle: bool = False,
        concurrent: bool = False,
        random_seed: int = 42,
        frame_ids: Optional[Iterable[FrameId]] = None,
        max_queue_size: int = 8,
        max_workers: int = 4,
    ) -> Generator[Frame[TDateTime], None, None]:
        runenv = pypeln.sync
        if concurrent:
            if not shuffle:
                raise ValueError("Order can not be guaranteed in concurrent mode!")

            runenv = pypeln.thread

        source_state = random.Random(random_seed)
        used_fids = [fid for fid in self.frame_ids if frame_ids is None or fid in frame_ids]
        if shuffle:
            source_state.shuffle(used_fids)

        yield from runenv.map(
            lambda fid: self.get_frame(frame_id=fid), used_fids, maxsize=max_queue_size, workers=max_workers
        )

    def sensor_pipeline(
        self,
        shuffle: bool = False,
        concurrent: bool = False,
        random_seed: int = 42,
        sensor_names: Optional[Iterable[SensorName]] = None,
        max_queue_size: int = 8,
        max_workers: int = 4,
    ) -> Generator[
        Union[
            CameraSensor[CameraSensorFrame[TDateTime]],
            LidarSensor[LidarSensorFrame[TDateTime]],
            RadarSensor[RadarSensorFrame[TDateTime]],
        ],
        None,
        None,
    ]:
        runenv = pypeln.sync
        if concurrent:
            if not shuffle:
                raise ValueError("Order can not be guaranteed in concurrent mode!")

            runenv = pypeln.thread

        source_state = random.Random(random_seed)
        used_sensors = [name for name in self.sensor_names if sensor_names is None or name in sensor_names]
        if shuffle:
            source_state.shuffle(used_sensors)

        yield from runenv.map(
            lambda name: self.get_sensor(sensor_name=name), used_sensors, maxsize=max_queue_size, workers=max_workers
        )

    def sensor_frame_pipeline(
        self,
        shuffle: bool = False,
        concurrent: bool = False,
        random_seed: int = 42,
        frame_ids: Optional[Iterable[FrameId]] = None,
        sensor_names: Optional[Iterable[SensorName]] = None,
        max_queue_size: int = 8,
        max_workers: int = 4,
        only_cameras: bool = False,
        only_radars: bool = False,
        only_lidars: bool = False,
    ) -> Generator[Tuple[SensorFrame[Optional[datetime]], Frame], None, None]:
        just_one = only_cameras ^ only_radars ^ only_lidars
        all_false = not (only_cameras or only_radars or only_lidars)
        if not just_one and not all_false:
            raise ValueError(
                "You can only set one of only_cameras or only_radars or "
                "only_lidars to a value to filter a certain sensor type!"
            )

        runenv = pypeln.sync
        if shuffle:
            runenv = pypeln.thread
        source_state = random.Random(random_seed)

        stage = self.frame_pipeline(
            shuffle=shuffle,
            concurrent=concurrent,
            random_seed=random_seed,
            frame_ids=frame_ids,
        )

        def map_frame(frame: Frame[TDateTime]):
            if only_cameras:
                used_sensor_names = frame.camera_names
            elif only_lidars:
                used_sensor_names = frame.lidar_names
            elif only_radars:
                used_sensor_names = frame.radar_names
            else:
                used_sensor_names = frame.sensor_names
            used_sensor_names = [name for name in used_sensor_names if sensor_names is None or name in sensor_names]
            if shuffle:
                source_state.shuffle(used_sensor_names)
            for name in used_sensor_names:
                yield frame.get_sensor(sensor_name=name), frame, self

        if not shuffle:
            yield from runenv.flat_map(map_frame, stage, maxsize=max_queue_size, workers=max_workers)
        else:
            yield from nested_generator_random_draw(
                source_generator=stage, nested_generator_factory=map_frame, endless_loop=False, random_seed=random_seed
            )

    def clear_from_cache(self):
        self._decoder.clear_from_cache()
