import os
from tempfile import TemporaryDirectory

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.v1.dataset import DGPDatasetEncoder
from paralleldomain.encoding.dgp.v1.pipeline_encoder import DGPV1DatasetPipelineEncoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities.any_path import AnyPath


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)
            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                workers_per_step=3,
                max_queue_size_per_step=12,
                set_stop=2,
                fs_copy=True,
                allowed_frames=[str(i) for i in range(num_frames)],
                output_annotation_types=[
                    AnnotationTypes.BoundingBoxes2D,
                    AnnotationTypes.SemanticSegmentation2D,
                    AnnotationTypes.Depth,
                    AnnotationTypes.OpticalFlow,
                    # AnnotationTypes.BoundingBoxes3D,
                    AnnotationTypes.InstanceSegmentation2D,
                ],
            )
            encoder.encode_dataset()

            assert (output_path / "scene_dataset.json").exists()
            for scene_name in dataset.scene_names:
                scene_folder = output_path / scene_name
                scene = dataset.get_scene(scene_name=scene_name)
                assert scene_folder.exists()
                assert any(
                    [
                        True
                        for f in (output_path / scene_name).iterdir()
                        if f.name.endswith(".json") and f.name.startswith("scene_")
                    ]
                )
                assert (scene_folder / DirectoryName.CALIBRATION).exists()
                assert len(list((scene_folder / DirectoryName.CALIBRATION).iterdir())) == 1
                assert (scene_folder / DirectoryName.ONTOLOGY).exists()
                assert len(list((scene_folder / DirectoryName.ONTOLOGY).iterdir())) == 1
                for sensor_name in scene.sensor_names:
                    assert (scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).exists()
                    assert (
                        len(list((scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).iterdir())) == num_frames
                    )
                    assert (scene_folder / DirectoryName.DEPTH / sensor_name).exists()
                    assert len(list((scene_folder / DirectoryName.DEPTH / sensor_name).iterdir())) == num_frames
                    assert (scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).exists()
                    assert (
                        len(list((scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).iterdir()))
                        == num_frames
                    )
                    assert (scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).exists()
                    assert (
                        len(list((scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).iterdir()))
                        == num_frames
                    )
                    assert (scene_folder / DirectoryName.RGB / sensor_name).exists()
                    assert len(list((scene_folder / DirectoryName.RGB / sensor_name).iterdir())) == num_frames
                    assert (scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).exists()
                    assert (
                        len(list((scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).iterdir()))
                        == num_frames
                    )
