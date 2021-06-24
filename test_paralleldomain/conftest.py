import os

import pytest

from paralleldomain import Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.model.dataset import Dataset


@pytest.fixture()
def decoder() -> Decoder:
    return DGPDecoder(dataset_path=os.environ["DATASET_PATH"])


@pytest.fixture()
def dataset(decoder: Decoder) -> Dataset:
    return Dataset.from_decoder(decoder=decoder)


@pytest.fixture()
def scene(dataset: Dataset) -> Scene:
    return dataset.get_scene(scene_name=dataset.scene_names[0])
