from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.model.unordered_scene import UnorderedScene

if TYPE_CHECKING:
    from paralleldomain.decoding.decoder import SceneDecoder

T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class SceneAccessDecoder(LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        settings: Optional[DecoderSettings],
        scene_name: SceneName,
        is_unordered_scene: bool,
        scene_decoder: SceneDecoder,
    ):
        if settings is None:
            settings = DecoderSettings()
        self.settings = settings
        self.dataset_name = dataset_name
        self.is_unordered_scene = is_unordered_scene
        self.scene_name = scene_name
        self.scene_decoder = scene_decoder

    def get_scene(self) -> Scene:
        _unique_cache_key = create_cache_key(scene_name=self.scene_name, extra="scene", dataset_name=self.dataset_name)
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_scene(),
        )

    def get_unordered_scene(self) -> UnorderedScene:
        if not self.is_unordered_scene:
            return self.get_scene()

        _unique_cache_key = create_cache_key(scene_name=self.scene_name, extra="scene", dataset_name=self.dataset_name)
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_unordered_scene(),
        )

    def _decode_scene(self) -> Scene:
        scene = Scene(decoder=self.scene_decoder)
        if self.settings.model_decorator is not None:
            scene = self.settings.model_decorator(scene)
        return scene

    def _decode_unordered_scene(self) -> UnorderedScene:
        scene = UnorderedScene(decoder=self.scene_decoder)
        if self.settings.model_decorator is not None:
            scene = self.settings.model_decorator(scene)
        return scene
