import contextlib
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, List

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import logging

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.scene import Scene, SceneDecoderProtocol
from paralleldomain.model.type_aliases import SceneName

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    """*Dataclass*

    Stores name, annotation types and any custom meta attributes for a dataset"""

    name: str
    """Dataset name."""
    available_annotation_types: List[AnnotationType]
    """List of all available annotation types in this dataset."""
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of custom attributes for the dataset."""


class DatasetDecoderProtocol(SceneDecoderProtocol, Protocol):
    """Base class for decoder implementations.

    Not to be instantiated directly!
    """

    def decode_scene_names(self) -> List[SceneName]:
        """Decodes scene names

        Returns:
            List of scene names

        """
        pass

    def decode_dataset_meta_data(self) -> DatasetMeta:
        """Decodes dataset metadata

        Returns:
            Dataset metadata

        """
        pass


class Dataset:
    """The :obj:`Dataset` object is the entry point for loading any data.

    A dataset manages all attached scenes and its sensor data. It takes care of calling the decoder when specific
    data is required and stores it in the PD SDK model classes and attributes.
    """

    def __init__(self, meta_data: DatasetMeta, scene_names: List[SceneName], decoder: DatasetDecoderProtocol):
        self._decoder = decoder
        self._scenes: Dict[SceneName, Scene] = dict()
        self.meta_data = meta_data
        self._scene_names: List[SceneName] = scene_names

    def _load_scene(self, scene_name: SceneName):
        if scene_name not in self._scenes:
            self._scenes[scene_name] = Scene.from_decoder(
                scene_name=scene_name, decoder=self._decoder, available_annotation_types=self.available_annotation_types
            )

    @property
    def scene_names(self) -> List[SceneName]:
        """Returns a list of scene names within the dataset."""
        return self._scene_names

    @property
    def scenes(self) -> Dict[SceneName, Scene]:
        """Returns a dictionary of :obj:`Scene` instances with the scene name as key."""
        for scene_name in self._scene_names:
            self._load_scene(scene_name=scene_name)
        return self._scenes

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        """Returns a list of available annotation types for the dataset."""
        return self.meta_data.available_annotation_types

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self.meta_data.name

    def get_scene(self, scene_name: SceneName) -> Scene:
        """Allows access to a scene by using its name.

        Args:
            scene_name (str): Name of scene to be returned

        Returns:
            Returns the `Scene` object for a scene name.
        """
        self._load_scene(scene_name=scene_name)
        return self._scenes[scene_name]

    @contextlib.contextmanager
    def get_editable_scene(self, scene_name: SceneName) -> ContextManager[Scene]:
        """[DEPRECATED]

        Model objects should not be edited directly, but only on copies (e.g., in an encoder).

        Args:
            scene_name (str): The scene name to use with locked cache.
        """
        with self.get_scene(scene_name=scene_name).editable() as scene:
            yield scene

    @staticmethod
    def from_decoder(decoder: DatasetDecoderProtocol) -> "Dataset":
        """Creates a dataset from any decoder following the `DatasetDecoderProtocol`

        Args:
            decoder (:class:`DatasetDecoderProtocol`): Instantiated implementation of the
                :class:`.DatasetDecoderProtocol`, e.g., an instance of `DGPDecoder`.

        Returns:
            Returns an instance of `Dataset`.

        """
        scene_names = decoder.decode_scene_names()
        meta_data = decoder.decode_dataset_meta_data()
        return Dataset(meta_data=meta_data, scene_names=scene_names, decoder=decoder)
