import os
from tempfile import TemporaryDirectory

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.encoder import DGPDatasetEncoder
from paralleldomain.utilities.any_path import AnyPath


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        output_path = AnyPath(TemporaryDirectory().name)

        # encoding
        encoder = DGPDatasetEncoder.from_dataset(dataset=dataset, output_path=output_path)
        encoder.run()

        # decoding the encoded
        decoder = DGPDecoder(dataset_path=output_path)
        Dataset.from_decoder(decoder=decoder)
