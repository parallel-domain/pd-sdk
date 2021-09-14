import os
from tempfile import TemporaryDirectory

from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.dataset import DGPDatasetEncoder
from paralleldomain.model.dataset import Dataset


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as output_path:
            # encoding
            encoder = DGPDatasetEncoder.from_dataset(dataset=dataset, output_path=output_path)
            encoder.encode_dataset()

            # decoding the encoded
            decoder = DGPDatasetDecoder(dataset_path=output_path)
            dataset = decoder.get_dataset()
