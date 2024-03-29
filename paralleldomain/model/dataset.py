import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Protocol, Set, Tuple, Type, TypeVar, Union

import pypeln

from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType
from paralleldomain.model.frame import Frame
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.generator_shuffle import nested_generator_random_draw

logger = logging.getLogger(__name__)
T = TypeVar("T")
TOrderBy = TypeVar("TOrderBy")
DEFAULT_NUM_OF_COUNTING_WORKERS = 4
DEFAULT_SIZE_OF_COUNTING_QUEUE = 8


@dataclass
class DatasetMeta:
    """Stores name, annotation types and any custom meta attributes for a dataset

    Args:
        name: :attr:`~.DatasetMeta.name`
        available_annotation_identifiers: :attr:`~.DatasetMeta.available_annotation_identifiers`
        custom_attributes: :attr:`~.DatasetMeta.custom_attributes`

    Attributes:
        name: Name of the dataset.
        available_annotation_identifiers: List of available annotation identifiers for all scenes.
        custom_attributes: Dictionary of arbitrary dataset attributes.

    """

    name: str
    available_annotation_identifiers: List[AnnotationIdentifier]
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        return list({a.annotation_type for a in self.available_annotation_identifiers})


class DatasetDecoderProtocol(Protocol):
    """Interface Definition for decoder implementations.

    Not to be instantiated directly!
    """

    def get_unordered_scene_names(self) -> List[SceneName]:
        pass

    def get_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        pass

    def get_dataset_metadata(self) -> DatasetMeta:
        pass

    def get_scene_names(self) -> List[SceneName]:
        pass

    def get_scene(self, scene_name: SceneName) -> Scene:
        pass

    def get_format(self) -> str:
        pass

    def get_path(self) -> Optional[AnyPath]:
        pass

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        pass


class Dataset:
    """The :obj:`Dataset` object is the entry point for loading any data.

    A dataset manages all attached scenes and its sensor data. It takes care of calling the decoder when specific
    data is required and stores it in the PD SDK model classes and attributes.
    """

    def __init__(self, decoder: DatasetDecoderProtocol):
        self._decoder = decoder
        self._number_of_camera_frames = None
        self._number_of_lidar_frames = None
        self._number_of_radar_frames = None
        self._number_of_sensor_frames = None

    def _get_unordered_scene_names(self) -> List[SceneName]:
        """Returns a list of sensor frame set names within the dataset."""
        return self._decoder.get_unordered_scene_names()

    @property
    def unordered_scene_names(self) -> List[SceneName]:
        """Returns a list of sensor frame set names within the dataset."""
        return self._get_unordered_scene_names()

    def _get_metadata(self) -> DatasetMeta:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_dataset_metadata()

    @property
    def metadata(self) -> DatasetMeta:
        """Returns a list of scene names within the dataset."""
        return self._get_metadata()

    @property
    def format(self) -> str:
        """Returns a str with the name of the dataset storage format (e.g. dgp, cityscapes, nuscenes)."""
        return self._decoder.get_format()

    @property
    def path(self) -> AnyPath:
        """Returns an optional path to a file or a folder where the dataset is stored."""
        return self._decoder.get_path()

    @property
    def decoder_init_kwargs(self) -> Dict[str, Any]:
        """Returns an optional path to a file or a folder where the dataset is stored."""
        return self._decoder.get_decoder_init_kwargs()

    @property
    def unordered_scenes(self) -> Dict[SceneName, UnorderedScene[Union[datetime, None]]]:
        """Returns a dictionary of :obj:`SensorFrameSet` instances with the scene name as key. This a superset of
        scenes and hence will also include all scenes (Scenes that have sensor frames with a date time)."""
        return {name: self._decoder.get_unordered_scene(scene_name=name) for name in self.unordered_scene_names}

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        """Returns a list of available annotation types for the dataset."""
        return self.metadata.available_annotation_types

    @property
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        """Returns a list of available annotation identifiers for the dataset."""
        return self.metadata.available_annotation_identifiers

    def get_annotation_identifiers_of_type(self, annotation_type: Type[T]) -> List[AnnotationIdentifier[T]]:
        return [
            identifier
            for identifier in self.available_annotation_identifiers
            if issubclass(identifier.annotation_type, annotation_type)
        ]

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self.metadata.name

    def get_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        """Allows access to a sensor frame set by using its name.

        Args:
            scene_name: Name of sensor frame set to be returned

        Returns:
            Returns the `SensorFrameSet` object for a sensor frame set name.
        """
        return self._decoder.get_unordered_scene(scene_name=scene_name)

    def _get_scene_names(self) -> List[SceneName]:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_scene_names()

    @property
    def scene_names(self) -> List[SceneName]:
        """Returns a list of scene names within the dataset."""
        return self._get_scene_names()

    @property
    def scenes(self) -> Dict[SceneName, Scene]:
        """Returns a dictionary of :obj:`Scene` instances with the scene name as key."""
        return {name: self._decoder.get_scene(scene_name=name) for name in self.scene_names}

    def get_scene(self, scene_name: SceneName) -> Scene:
        """Allows access to a scene by using its name.

        Args:
            scene_name: Name of scene to be returned

        Returns:
            Returns the :obj:`Scene` object for a scene name.
        """
        return self._decoder.get_scene(scene_name=scene_name)

    def get_sensor_frames(
        self,
        sensor_names: Optional[Iterable[SensorName]] = None,
        frame_ids: Optional[Iterable[FrameId]] = None,
        scene_names: Optional[Iterable[SceneName]] = None,
        shuffle: bool = False,
        concurrent: bool = False,
        max_queue_size: int = 8,
        max_workers: int = 4,
        random_seed: int = 42,
        only_cameras: bool = False,
        only_radars: bool = False,
        only_lidars: bool = False,
    ) -> Generator[SensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all sensor frames from the scenes with the given names.
        If None is passed for scenes_names all scenes in this dataset are used. Same for sensor_names and frame_ids.
        None means all available will be returned.
        """
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
            scene_names=scene_names,
        ):
            yield sensor_frame

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all CameraSensorFrames of all the unordered scenes in this dataset.
        """
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_cameras=True):
            yield sensor_frame

    @property
    def camera_names(self) -> Set[SensorName]:
        """
        Returns the names of all camera sensors across all scenes in this dataset.
        """
        stage = self.scene_pipeline(shuffle=True, concurrent=True)
        stage = pypeln.thread.flat_map(
            lambda scene: scene.camera_names,
            stage,
            maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
            workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
        )

        names = set()
        for camera_names in stage:
            names.add(camera_names)
        return names

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all LidarSensorFrames of all the unordered scenes in this dataset.
        """
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_lidars=True):
            yield sensor_frame

    @property
    def lidar_names(self) -> Set[SensorName]:
        """
        Returns the names of all lidar sensors across all scenes in this dataset.
        """
        stage = self.scene_pipeline(shuffle=True, concurrent=True)
        stage = pypeln.thread.flat_map(
            lambda scene: scene.lidar_names,
            stage,
            maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
            workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
        )

        names = set()
        for lidar_names in stage:
            names.add(lidar_names)
        return names

    @property
    def radar_frames(self) -> Generator[RadarSensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all RadarSensorFrames of all the unordered scenes in this dataset.
        """
        for sensor_frame, _, _ in self.sensor_frame_pipeline(shuffle=False, only_radars=True):
            yield sensor_frame

    @property
    def radar_names(self) -> Set[SensorName]:
        """
        Returns the names of all Radar sensors across all scenes in this dataset.
        """
        stage = self.scene_pipeline(shuffle=True, concurrent=True)
        stage = pypeln.thread.flat_map(
            lambda scene: scene.radar_names,
            stage,
            maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
            workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
        )

        names = set()
        for radar_names in stage:
            names.add(radar_names)
        return names

    @property
    def sensor_frames(self) -> Generator[SensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all SensorFrames (Lidar and Camera) of all the unordered scenes in this dataset.
        """
        for sf, _, _ in self.sensor_frame_pipeline(shuffle=False):
            yield sf

    @property
    def number_of_camera_frames(self) -> int:
        if self._number_of_camera_frames is None:
            self._number_of_camera_frames = 0
            stage = pypeln.sync.from_iterable(self.scene_pipeline(shuffle=True, concurrent=True))
            stage = pypeln.thread.map(
                lambda scene: scene.number_of_camera_frames,
                stage,
                maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
                workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
            )

            for cnt in stage:
                self._number_of_camera_frames += cnt
        return self._number_of_camera_frames

    @property
    def number_of_lidar_frames(self) -> int:
        if self._number_of_lidar_frames is None:
            self._number_of_lidar_frames = 0
            stage = pypeln.sync.from_iterable(self.scene_pipeline(shuffle=True, concurrent=True))
            stage = pypeln.thread.map(
                lambda scene: scene.number_of_lidar_frames,
                stage,
                maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
                workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
            )

            for cnt in stage:
                self._number_of_lidar_frames += cnt
        return self._number_of_lidar_frames

    @property
    def number_of_radar_frames(self) -> int:
        if self._number_of_radar_frames is None:
            self._number_of_radar_frames = 0
            stage = pypeln.sync.from_iterable(self.scene_pipeline(shuffle=True, concurrent=True))
            stage = pypeln.thread.map(
                lambda scene: scene.number_of_radar_frames,
                stage,
                maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
                workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
            )

            for cnt in stage:
                self._number_of_radar_frames += cnt
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
                stage = pypeln.sync.from_iterable(self.scene_pipeline(shuffle=True, concurrent=True))
                stage = pypeln.thread.map(
                    lambda scene: scene.number_of_sensor_frames,
                    stage,
                    maxsize=DEFAULT_SIZE_OF_COUNTING_QUEUE,
                    workers=DEFAULT_NUM_OF_COUNTING_WORKERS,
                )

                for cnt in stage:
                    self._number_of_sensor_frames += cnt
        return self._number_of_sensor_frames

    def scene_pipeline(
        self,
        shuffle: bool = False,
        concurrent: bool = False,
        endless_loop: bool = False,
        random_seed: int = 42,
        scene_names: Optional[List[SceneName]] = None,
        only_ordered_scenes: bool = False,
        max_queue_size: int = 8,
        max_workers: int = 4,
    ) -> Generator[Union[UnorderedScene, Scene], None, None]:
        """
        Returns a generator that yields all scenes from the dataset with the given names.
        If None is passed for scenes_names all scenes in this dataset are returned.
        """
        runenv = pypeln.sync
        if concurrent:
            if not shuffle:
                raise ValueError("Order can not be guaranteed in concurrent mode!")
            runenv = pypeln.thread

        def _source_loop():
            source_state = random.Random(random_seed)
            used_scene_names = self.scene_names if only_ordered_scenes else self.unordered_scene_names
            used_scene_names = [name for name in used_scene_names if scene_names is None or name in scene_names]
            epoch = 0
            if len(used_scene_names) == 0:
                raise ValueError("scene_names is empty.")
            while endless_loop or epoch == 0:
                epoch += 1
                if shuffle:
                    source_state.shuffle(used_scene_names)
                yield from used_scene_names

        def _scene_decoding(scene_name: SceneName) -> Union[UnorderedScene, Scene]:
            if only_ordered_scenes:
                scene = self.get_scene(scene_name=scene_name)
            else:
                scene = self.get_unordered_scene(scene_name=scene_name)
            return scene

        yield from runenv.map(_scene_decoding, _source_loop(), maxsize=max_queue_size, workers=max_workers)

    def sensor_frame_pipeline(
        self,
        shuffle: bool = False,
        concurrent: bool = False,
        endless_loop: bool = False,
        random_seed: int = 42,
        scene_names: Optional[List[SceneName]] = None,
        frame_ids: Optional[Union[Dict[SceneName, Iterable[FrameId]], Iterable[FrameId]]] = None,
        sensor_names: Optional[Iterable[SensorName]] = None,
        only_ordered_scenes: bool = False,
        max_queue_size: int = 8,
        max_workers: int = 4,
        only_cameras: bool = False,
        only_radars: bool = False,
        only_lidars: bool = False,
    ) -> Generator[Tuple[SensorFrame[Optional[datetime]], Frame, Union[UnorderedScene, Scene]], None, None]:
        """
        Returns a generator that yields tuples of SensorFrame, Frame, Scene from the dataset.
        It can also be configured to only return certain scenes, frames, sensors.
        By setting shuffle = True the data is returned a randomized order.
        By setting concurrent = True threads are used to speed up the generator.

        You can use this as quick access to sensor frames to load images, annotations etc. and feed those to
        your models or to encode the sensor frames into another data format.

        Args:
            shuffle: if = True the data is returned a randomized order
            concurrent: if = True threads are used to speed up the generator
            endless_loop: whether to iterate over the scenes of the dataset in an endless loop
            random_seed: seed used for shuffling the data
            scene_names: optional iterable of scene_names, if set, only frames from these scenes are yielded
            frame_ids: optional, dict/iterable containing frame_ids::
                       - if frame_ids is None, yield all frames from each scene
                       - if frame_ids is a dictionary, mapping scene_names to frame_ids:
                           + if a scene_name is among the dictionary keys, the corresponding value defines which
                             frames to yield from that scene
                           + if a scene_name is not in the dictionary, yield all frames from that scene
                       - if frame_ids is an iterable, yield the same set of frames from each scene
            sensor_names: optional iterable of sensor_names, if set, only frames recorded with these sensors are yielded
            only_ordered_scenes: if = True, only yield frames from ordered scenes
            max_queue_size: maximum queue size
            max_workers: max number of worker threads
            only_cameras: only yield camera frames
            only_radars: only yield radar frames
            only_lidars: only yield lidar frames
        """
        runenv = pypeln.sync
        if concurrent:
            if not shuffle:
                raise ValueError("Order can not be guaranteed in concurrent mode!")
            runenv = pypeln.thread
        source_state = random.Random(random_seed)

        stage = self.scene_pipeline(
            shuffle=shuffle,
            concurrent=concurrent,
            # endless looping is differently handled in shuffle pipeline
            endless_loop=endless_loop and not shuffle,
            random_seed=source_state.randint(0, 99999),
            scene_names=scene_names,
            only_ordered_scenes=only_ordered_scenes,
        )

        def map_scenes(scene: Union[UnorderedScene, Scene]):
            def get_relevant_frame_ids(scene_name):
                # if frame_ids is None, we can yield all frames from the scene
                if frame_ids is None:
                    return None
                # if frame_ids is a dictionary, and it contains the scene_name as a key,
                # we should only yield the frame id-s defined in the corresponding value
                elif isinstance(frame_ids, dict):
                    return frame_ids.get(scene_name, None)
                # if frame_ids is an iterable, i.e. there is no mapping to specific scenes,
                # yield the same set of frames for all scenes
                return frame_ids

            yield from scene.sensor_frame_pipeline(
                frame_ids=get_relevant_frame_ids(scene.name),
                sensor_names=sensor_names,
                random_seed=source_state.randint(0, 99999),
                max_workers=max_workers,
                max_queue_size=max_queue_size,
                shuffle=shuffle,
                concurrent=concurrent,
                only_cameras=only_cameras,
                only_radars=only_radars,
                only_lidars=only_lidars,
            )

        if not shuffle:
            stage = runenv.flat_map(map_scenes, stage, maxsize=max_queue_size, workers=max_workers)
            yield from pypeln.sync.to_iterable(stage, maxsize=max_queue_size)
        else:
            yield from nested_generator_random_draw(
                source_generator=stage,
                nested_generator_factory=map_scenes,
                endless_loop=endless_loop,
                random_seed=random_seed,
            )
