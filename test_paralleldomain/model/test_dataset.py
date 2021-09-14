from paralleldomain.decoding.decoder import DatasetDecoder


def test_can_load_dataset_from_path(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    assert len(dataset.scene_names) > 0


def test_can_load_scene(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    scene = dataset.get_scene(scene_name=dataset.scene_names[0])
    assert scene is not None
    assert scene.name == dataset.scene_names[0]
