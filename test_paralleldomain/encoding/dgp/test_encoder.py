from tempfile import TemporaryDirectory

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.encoder import DGPEncoder
from paralleldomain.model.annotation import AnnotationTypes


def test_encoding_of_modified_scene(dataset: Dataset):
    with dataset.get_editable_scene(scene_name=dataset.scene_names[0]) as scene:
        cam = scene.cameras[0]
        sensor_frame = cam.get_frame(frame_id=scene.frame_ids[0])
        semseg = sensor_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        semseg.class_ids[...] = 1337

        with TemporaryDirectory() as temp_dir:
            encoder = DGPEncoder(dataset_path=temp_dir)
            encoder.encode_scene(scene=scene)

            decoder = DGPDecoder(dataset_path=temp_dir)
            decoded_dataset = Dataset.from_decoder(decoder=decoder)
            decoded_scene = decoded_dataset.get_scene(scene_name=scene.name)
            assert scene.sensor_names == decoded_scene.sensor_names
            # TODO more checks
