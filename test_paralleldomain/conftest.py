import pytest

from paralleldomain import Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_decoder import DGPDecoder
from paralleldomain.model.dataset import Dataset


@pytest.fixture()
def decoder() -> Decoder:
    return DGPDecoder(dataset_path="replace this with a url to test!")


@pytest.fixture()
def dataset(decoder: Decoder) -> Dataset:
    return Dataset.from_decoder(decoder=decoder)


@pytest.fixture()
def scene(dataset: Dataset) -> Scene:
    return dataset.get_scene(scene_name=dataset.scene_names[0])
