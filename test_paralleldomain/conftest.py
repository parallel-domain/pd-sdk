import pytest
from paralleldomain import Scene

from paralleldomain.model.dataset import Dataset
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_decoder import DGPDecoder


@pytest.fixture()
def decoder() -> Decoder:
    return DGPDecoder(dataset_path="replace this with a s3 or local path to a dgp!")


@pytest.fixture()
def scene(decoder: Decoder) -> Scene:
    dataset = Dataset.from_decoder(decoder=decoder)
    return dataset.get_scene(scene_name=dataset.scene_names[0])
