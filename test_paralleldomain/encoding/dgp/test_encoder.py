import os
from tempfile import TemporaryDirectory

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.dataset import DGPDatasetEncoder
from paralleldomain.utilities.any_path import AnyPath


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as output_path:
            # encoding
            encoder = DGPDatasetEncoder.from_dataset(dataset=dataset, output_path=output_path)
            encoder.encode_dataset()

            # decoding the encoded
            decoder = DGPDecoder(dataset_path=output_path)
            Dataset.from_decoder(decoder=decoder)
