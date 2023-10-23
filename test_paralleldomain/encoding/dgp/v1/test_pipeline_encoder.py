import os
from tempfile import TemporaryDirectory
from typing import List, Type

import pytest

from paralleldomain import Scene
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.v1.dataset import DGPDatasetEncoder
from paralleldomain.encoding.dgp.v1.encoding_format import DGPV1EncodingFormat
from paralleldomain.encoding.dgp.v1.pipeline_encoder import DGPV1DatasetPipelineEncoder
from paralleldomain.encoding.helper import encode_dataset
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.sensor import FilePathedDataType
from paralleldomain.model.type_aliases import FrameId
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.dataset_transform import DatasetTransformation, DataTransformer


def check_if_files_exist(
    output_path: AnyPath,
    dataset: Dataset,
    scene_names: List[str],
    camera_names: List[str],
    lidar_names: List[str],
    num_frames: int,
    num_calibration_files: int = 1,
):
    assert (output_path / "scene_dataset.json").exists()
    for scene_name in scene_names:
        scene_folder = output_path / scene_name
        assert scene_folder.exists()
        assert any(
            [
                True
                for f in (output_path / scene_name).iterdir()
                if f.name.endswith(".json") and f.name.startswith("scene_")
            ]
        )
        assert (scene_folder / DirectoryName.CALIBRATION).exists()
        assert len(list((scene_folder / DirectoryName.CALIBRATION).iterdir())) == num_calibration_files
        assert (scene_folder / DirectoryName.ONTOLOGY).exists()
        assert len(list((scene_folder / DirectoryName.ONTOLOGY).iterdir())) == 1
        for sensor_name in camera_names:
            assert (scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.DEPTH / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.DEPTH / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).exists()
            assert (
                len(list((scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).iterdir())) == num_frames
            )
            assert (scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).exists()
            assert (
                len(list((scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).iterdir())) == num_frames
            )
            assert (scene_folder / DirectoryName.RGB / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.RGB / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).iterdir())) == num_frames

        for sensor_name in lidar_names:
            assert (scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).iterdir())) == num_frames


def _encode(num_frames: int, dataset: Dataset, output_path: AnyPath, annotation_types: List[Type], **kwargs):
    class FilterFrames(DataTransformer[List[FrameId], Scene]):
        def __call__(self, model: Scene) -> List[FrameId]:
            return model.frame_ids[:num_frames]

    data_transform = DatasetTransformation(
        frame_ids=FilterFrames(),
        sensor_names=["CAM_BACK", "CAM_FRONT"],
        scene_names=dataset.scene_names[:2],
    )
    dataset = data_transform.apply(dataset)
    encode_dataset(
        dataset=dataset,
        encoding_format=DGPV1EncodingFormat(
            dataset_output_path=output_path,
            target_dataset_name="NewDataset",
        ),
        copy_data_types=annotation_types,
        workers=4,
        max_in_queue_size=10,
        **kwargs,
    )


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)
            annotation_types = dataset.available_annotation_types
            annotation_types = list(
                set(annotation_types).intersection(
                    {
                        FilePathedDataType.BoundingBoxes2D,
                        FilePathedDataType.SemanticSegmentation2D,
                        FilePathedDataType.Depth,
                        FilePathedDataType.OpticalFlow,
                        FilePathedDataType.BoundingBoxes3D,
                        FilePathedDataType.InstanceSegmentation2D,
                    }
                )
            )
            if len(dataset.camera_names) > 0:
                annotation_types.append(FilePathedDataType.Image)
            if len(dataset.lidar_names) > 0:
                annotation_types.append(FilePathedDataType.PointCloud)
            _encode(num_frames=num_frames, dataset=dataset, output_path=output_path, annotation_types=annotation_types)
            check_if_files_exist(
                output_path=output_path,
                dataset=dataset,
                scene_names=dataset.scene_names[:2],
                camera_names=list(dataset.camera_names),
                num_frames=num_frames,
                lidar_names=list(dataset.lidar_names),
            )


def test_should_copy_callback(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)
            only_copy_sensor_name = list(dataset.camera_names)[0]
            with pytest.raises(KeyError, match="rgb"):
                _encode(
                    num_frames=num_frames,
                    dataset=dataset,
                    output_path=output_path,
                    annotation_types=[
                        FilePathedDataType.BoundingBoxes2D,
                        FilePathedDataType.Image,
                        FilePathedDataType.SemanticSegmentation2D,
                        FilePathedDataType.Depth,
                        FilePathedDataType.OpticalFlow,
                        FilePathedDataType.BoundingBoxes3D,
                        FilePathedDataType.InstanceSegmentation2D,
                    ],
                    should_copy_callbacks={
                        FilePathedDataType.Image: lambda d, s: s.sensor_name == only_copy_sensor_name,
                    },
                )
