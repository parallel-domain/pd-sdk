import pytest
from paralleldomain import Scene

from paralleldomain.model.dataset import Dataset
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_decoder import DGPDecoder


@pytest.fixture()
def decoder() -> Decoder:
    return DGPDecoder(dataset_path="s3://paralleldomain-silo-ai/public_pdviz_6/")


@pytest.fixture()
def scene(decoder: Decoder) -> Scene:
    dataset = Dataset.from_decoder(decoder=decoder)
    return dataset.get_scene(scene_name=dataset.scene_names[0])
