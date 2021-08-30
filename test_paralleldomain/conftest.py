import os

import pytest

from paralleldomain import Scene
from paralleldomain.decoding.decoder import Decoder, TemporalDecoder
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.model.dataset import Dataset, SceneDataset
from paralleldomain.model.sensor import TemporalSensorFrame


@pytest.fixture()
def decoder() -> TemporalDecoder[SceneDataset, TemporalSensorFrame]:
    return DGPDecoder(dataset_path=os.environ["DATASET_PATH"])


@pytest.fixture()
def dataset(decoder: TemporalDecoder[SceneDataset, TemporalSensorFrame]) -> SceneDataset:
    return decoder.get_dataset()


@pytest.fixture()
def scene(dataset: SceneDataset) -> Scene:
    return dataset.get_scene(scene_name=list(dataset.scene_names)[0])
