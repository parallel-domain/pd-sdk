from tempfile import TemporaryDirectory

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.encoder import DGPEncoder


def test_encoding_of_modified_scene(dataset: Dataset):
    """
    decoder = DGPDecoder(dataset_path=dataset_input_path)
    dataset = Dataset.from_decoder(decoder=decoder)

    with DGPEncoder(dataset=dataset, output_path=dataset_output_path, frame_slicing=slice(0, 2)) as encoder:
        with dataset.get_editable_scene(scene_name=dataset.scene_names[0]) as scene:
            encoder.encode_scene(scene)



    with dataset.get_editable_scene(scene_name=dataset.scene_names[0]) as scene:
        with TemporaryDirectory() as temp_dir:
            encoder = DGPEncoder(output_path=temp_dir)
            encoder.encode_dataset(dataset)

            decoder = DGPDecoder(dataset_path=temp_dir)
            decoded_dataset = Dataset.from_decoder(decoder=decoder)
            decoded_scene = decoded_dataset.get_scene(scene_name=scene.name)"""
