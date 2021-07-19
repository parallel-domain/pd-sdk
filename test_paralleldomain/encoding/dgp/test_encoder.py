import os
from tempfile import TemporaryDirectory

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.encoder import DGPEncoder
from paralleldomain.utilities.any_path import AnyPath


def test_encoding_of_modified_scene(dataset: Dataset):
    if not os.environ["SKIP_ENCODER"]:
        output_path = AnyPath(TemporaryDirectory().name)

        # encoding
        with DGPEncoder(dataset=dataset, output_path=output_path, frame_slice=slice(0, 2)) as encoder:
            with dataset.get_editable_scene(scene_name=dataset.scene_names[0]) as scene:
                encoder.encode_scene(scene)

        # decoding the encoded
        decoder = DGPDecoder(dataset_path=output_path)
        Dataset.from_decoder(decoder=decoder)
