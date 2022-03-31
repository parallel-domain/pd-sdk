from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar, Union

from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import logging

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.scene import Scene
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

logger = logging.getLogger(__name__)
TOrderBy = TypeVar("TOrderBy")


@dataclass
class DatasetMeta:
    """Stores name, annotation types and any custom meta attributes for a dataset

    Args:
        name: :attr:`~.DatasetMeta.name`
        available_annotation_types: :attr:`~.DatasetMeta.available_annotation_types`
        custom_attributes: :attr:`~.DatasetMeta.custom_attributes`

    Attributes:
        name: Name of the dataset.
        available_annotation_types: List of available annotation types for all scenes.
        custom_attributes: Dictionary of arbitrary dataset attributes.

    """

    name: str
    available_annotation_types: List[AnnotationType]
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


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

    @property
    def unordered_scene_names(self) -> List[SceneName]:
        """Returns a list of sensor frame set names within the dataset."""
        return self._decoder.get_unordered_scene_names()

    @property
    def metadata(self) -> DatasetMeta:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_dataset_metadata()

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

    @property
    def scene_names(self) -> List[SceneName]:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_scene_names()

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
    ) -> Generator[SensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yield all sensor frames from the scenes with the given names.
        If None is passed for scenes_names all scenes in this dataset are used. Same for sensor_names and frame_ids.
        None means all available will be returned.
        """
        if scene_names is None:
            scene_names = self.unordered_scene_names
        for scene_name in scene_names:
            scene = self.get_unordered_scene(scene_name=scene_name)
            yield from scene.get_sensor_frames(sensor_names=sensor_names, frame_ids=frame_ids)

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all CameraSensorFrames of all the unordered scenes in this dataset.
        """
        for scene in self.unordered_scenes.values():
            yield from scene.camera_frames

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all LidarSensorFrames of all the unordered scenes in this dataset.
        """
        for scene in self.unordered_scenes.values():
            yield from scene.lidar_frames

    @property
    def sensor_frames(self) -> Generator[SensorFrame[Optional[datetime]], None, None]:
        """
        Returns a generator that yields all SensorFrames (Lidar and Camera) of all the unordered scenes in this dataset.
        """
        yield from self.camera_frames
        yield from self.lidar_frames

    @property
    def number_of_camera_frames(self) -> int:
        if self._number_of_camera_frames is None:
            self._number_of_camera_frames = 0
            for scene in self.unordered_scenes.values():
                self._number_of_camera_frames += scene.number_of_camera_frames
        return self._number_of_camera_frames

    @property
    def number_of_lidar_frames(self) -> int:
        if self._number_of_lidar_frames is None:
            self._number_of_lidar_frames = 0
            for scene in self.unordered_scenes.values():
                self._number_of_lidar_frames += scene.number_of_lidar_frames
        return self._number_of_lidar_frames

    @property
    def number_of_sensor_frames(self) -> int:
        return self.number_of_lidar_frames + self.number_of_camera_frames
